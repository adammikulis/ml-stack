"""ccache in the llama.cpp source build: the cmake flags and the environment it gets, the one extra
directory the sandbox lets it write (and that nothing else opens up), and the line the build says
when ccache is missing. None of this compiles llama.cpp."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from poolhouse import home
from poolhouse.sandbox import run
from poolhouse.sandbox.seatbelt import Seatbelt
from poolhouse.serve import llamacpp_ccache, llamacpp_compile
from poolhouse.serve.build_paths import BuildFailed
from poolhouse.serve.llamacpp_compile import Job, Toolchain

CMAKE = """cmake_minimum_required(VERSION 3.14)
project(stub NONE)
add_custom_target(llama-server ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/server.sh ${CMAKE_BINARY_DIR}/bin/llama-server)
"""


@pytest.fixture
def seatbelt() -> Seatbelt:
    """The real Seatbelt backend, or a skip where ``sandbox-exec`` does not exist."""
    backend = Seatbelt()
    if not (state := backend.available()).ok:
        pytest.skip(state.reason)
    return backend


def toolchain() -> Toolchain:
    """The real toolchain of this machine, or a skip where a build cannot run at all."""
    try:
        return llamacpp_compile.toolchain()
    except BuildFailed as exc:
        pytest.skip(str(exc))


def present(tmp_path: Path) -> llamacpp_ccache.Ccache:
    return llamacpp_ccache.Ccache("/opt/ccache/bin/ccache", tmp_path / "cc", ("/opt/ccache",))


def tools_with(cached: llamacpp_ccache.Ccache | None) -> Toolchain:
    base = toolchain()
    return Toolchain(base.cmake, base.cc, base.cxx, base.sysroot,
                     tuple(sorted({*base.reads, *(cached.reads if cached else ())})), base.make, cached)


def test_the_cmake_flags_put_ccache_in_front_of_every_language_only_when_it_is_present(tmp_path):
    cached = present(tmp_path)
    with_it = tools_with(cached).arguments()
    without = tools_with(None).arguments()
    launchers = [f"-DCMAKE_{lang}_COMPILER_LAUNCHER={cached.exe}" for lang in ("C", "CXX", "OBJC", "OBJCXX")]
    assert all(flag in with_it for flag in launchers)
    assert [a for a in with_it if "LAUNCHER" not in a] == without
    assert not any("ccache" in a for a in without)


def test_the_environment_names_the_cache_its_limit_and_a_base_directory_above_the_work_dir(tmp_path):
    cached = present(tmp_path)
    env = cached.environment(tmp_path / "work" / "b100")
    assert env == {"CCACHE_DIR": str(tmp_path / "cc"), "CCACHE_MAXSIZE": "5G",
                   "CCACHE_BASEDIR": str(tmp_path / "work" / "b100"), "CCACHE_NOHASHDIR": "1"}
    assert llamacpp_ccache.MAX_SIZE == "5G"


def test_the_cache_lives_under_the_project_home_and_is_made_when_ccache_is_found(tmp_path, monkeypatch):
    stub = tmp_path / "bin" / "ccache"
    stub.parent.mkdir()
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(stub.parent))
    found = llamacpp_ccache.find()
    assert found is not None and found.exe == os.path.realpath(stub)
    assert found.directory.is_dir()
    assert found.directory == Path(os.path.realpath(home.state("llama.cpp", "ccache")))
    assert found.directory.is_relative_to(Path(os.path.realpath(home.home())))


def test_without_ccache_nothing_is_found_and_no_flag_or_variable_is_added(tmp_path, monkeypatch):
    tools = tools_with(None)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert llamacpp_ccache.find() is None
    work = tmp_path / "work" / "b1" / "scratch"
    policy = llamacpp_compile._policy(tools, tmp_path / "src", work, 60.0)
    assert policy.cache == "" and policy.write == (str(work),)
    assert not any(key.startswith("CCACHE") for key in policy.env)


def test_with_ccache_the_policy_adds_the_one_cache_directory_and_its_variables(tmp_path):
    cached = present(tmp_path)
    work = tmp_path / "work" / "b1" / "scratch"
    policy = llamacpp_compile._policy(tools_with(cached), tmp_path / "src", work, 60.0)
    assert policy.write == (str(work),)
    assert policy.cache == str(cached.directory)
    assert policy.env["CCACHE_DIR"] == str(cached.directory)
    assert policy.env["CCACHE_BASEDIR"] == str(work.parent)


def test_the_sandbox_writes_the_work_dir_and_the_cache_and_still_refuses_anywhere_else(tmp_path, seatbelt):
    cached = llamacpp_ccache.Ccache("/usr/bin/true", Path(os.path.realpath(tmp_path)) / "cc", ())
    cached.directory.mkdir()
    work = Path(os.path.realpath(tmp_path)) / "work" / "b1" / "scratch"
    work.mkdir(parents=True)
    elsewhere = Path(os.path.realpath(tmp_path)) / "elsewhere"
    elsewhere.mkdir()
    sibling = work.parent / "src"
    sibling.mkdir()
    policy = llamacpp_compile._policy(tools_with(cached), sibling, work, 60.0)

    def attempt(where: Path) -> int:
        return run(["/bin/sh", "-c", f"echo x > {where}/marker"], policy, cwd=str(work)).returncode

    assert attempt(work) == 0 and attempt(cached.directory) == 0
    assert attempt(elsewhere) != 0 and not (elsewhere / "marker").exists()
    assert attempt(work.parent) != 0 and not (work.parent / "marker").exists()
    assert attempt(cached.directory.parent) != 0 and not (cached.directory.parent / "marker").exists()
    assert attempt(Path(os.path.realpath(home.user_home()))) != 0


def test_the_real_ccache_runs_in_the_sandbox_and_hits_across_work_directories(tmp_path, seatbelt):
    if shutil.which("ccache") is None:
        pytest.skip("needs ccache")
    tools = toolchain()
    cached = tools.ccache
    assert cached is not None
    root = Path(os.path.realpath(tmp_path)) / "work"
    cached.reset(root / "b1")
    for name in ("b1", "b2"):
        work = root / name / "scratch"
        (root / name / "src").mkdir(parents=True)
        work.mkdir()
        source = root / name / "src" / "x.c"
        source.write_text("int answer(void) { return 42; }\n")
        policy = llamacpp_compile._policy(tools, source.parent, work, 120.0)
        done = run([cached.exe, tools.cc, "-isysroot", tools.sysroot, "-c", str(source), "-o", "x.o"]
                   if tools.sysroot else [cached.exe, tools.cc, "-c", str(source), "-o", "x.o"],
                   policy, cwd=str(work))
        assert done.returncode == 0, done.stderr
        assert (work / "x.o").is_file()
    line = cached.summary(root / "b2")
    assert line.startswith("ccache: 1 hits, 1 misses (50% hit rate)"), line


def test_the_summary_is_one_line_even_when_ccache_gives_nothing(tmp_path):
    silent = tmp_path / "ccache"
    silent.write_text("#!/bin/sh\nexit 1\n")
    silent.chmod(0o755)
    line = llamacpp_ccache.Ccache(str(silent), tmp_path / "cc", ()).summary(tmp_path)
    assert "\n" not in line and line == "ccache: statistics unavailable"


def test_a_build_without_ccache_says_so_in_one_line_and_still_builds(tmp_path, seatbelt):
    source = tmp_path / "work" / "b1" / "src"
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text(CMAKE)
    (source / "server.sh").write_text("#!/bin/sh\nexit 0\n")
    scratch = tmp_path / "work" / "b1" / "scratch"
    scratch.mkdir()
    said: list[str] = []
    built = llamacpp_compile.compile_source(Job(source, scratch, 1, "a" * 40, 2, tools_with(None)), said.append)
    assert built.is_file()
    assert said.count("  " + llamacpp_ccache.ABSENT) == 1
    assert not any("hits" in line for line in said)
    assert "writes only " + str(Path(os.path.realpath(scratch))) + ")" in " ".join(said)


def fake(directory: Path, body: str, name: str = "ccache") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def test_a_ccache_that_hangs_is_killed_and_the_build_goes_without_it(tmp_path, monkeypatch):
    fake(tmp_path / "bin", "sleep 300\n")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setattr(llamacpp_ccache, "SECONDS", 1.0)
    started = time.monotonic()
    assert llamacpp_ccache.find() is None
    assert time.monotonic() - started < 20
    assert llamacpp_ccache.ABSENT.count("\n") == 0


def test_a_statistics_call_that_hangs_gives_a_line_and_not_a_hang(tmp_path, monkeypatch):
    path = fake(tmp_path / "bin", 'case "$1" in --version) exit 0;; *) sleep 300;; esac\n')
    monkeypatch.setattr(llamacpp_ccache, "SECONDS", 1.0)
    started = time.monotonic()
    line = llamacpp_ccache.Ccache(str(path), tmp_path / "cc", ()).summary(tmp_path)
    assert line == "ccache: statistics unavailable" and time.monotonic() - started < 20


def test_a_flood_of_output_is_cut_off_and_non_utf8_bytes_do_not_crash_the_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(llamacpp_ccache, "OUTPUT_BYTES", 10_000)
    flood = fake(tmp_path / "a", "yes 'cache_miss\t5'\n")
    assert llamacpp_ccache.Ccache(str(flood), tmp_path / "cc", ()).summary(tmp_path) == "ccache: statistics unavailable"
    junk = fake(tmp_path / "b", "printf 'direct_cache_hit\\t3\\n\\377\\376garbage\\ncache_miss\\t1\\n'\n")
    line = llamacpp_ccache.Ccache(str(junk), tmp_path / "cc", ()).summary(tmp_path)
    assert line.startswith("ccache: 3 hits, 1 misses (75% hit rate)")


def test_a_statistics_call_that_fails_does_not_fail_the_build_line(tmp_path):
    bad = fake(tmp_path / "bin", "exit 3\n")
    line = llamacpp_ccache.Ccache(str(bad), tmp_path / "cc", ()).summary(tmp_path)
    assert line == "ccache: statistics unavailable"
    llamacpp_ccache.Ccache(str(bad), tmp_path / "cc", ()).reset(tmp_path)


def test_a_path_with_spaces_quotes_semicolons_and_a_newline_reaches_the_process_as_one_argument(tmp_path, monkeypatch):
    log = tmp_path / "log"
    odd = tmp_path / "it's a; dir $(touch pwned)\n`x`" / "bin"
    fake(odd, f'printf "%s|%s\\n" "$0" "$#" >> "{log}"\nexit 0\n')
    monkeypatch.setenv("PATH", str(odd))
    found = llamacpp_ccache.find()
    assert found is not None and found.exe == os.path.realpath(odd / "ccache")
    found.reset(tmp_path)
    lines = log.read_text().split("\n|")
    assert all(entry.rstrip().endswith("|1") or entry.rstrip().endswith("|2") for entry in lines if entry)
    assert not list(tmp_path.rglob("pwned")) and not Path("pwned").exists()


def test_library_listing_survives_a_non_binary_a_symlink_loop_and_an_unreadable_file(tmp_path):
    text = tmp_path / "text"
    text.write_text("not a binary\n")
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    locked = tmp_path / "locked"
    locked.write_text("x")
    locked.chmod(0)
    try:
        for target in (text, loop, locked, tmp_path / "missing", Path("/odd 'name'; \n")):
            assert isinstance(llamacpp_ccache._libraries(str(target)), list)
            assert isinstance(llamacpp_ccache._tree(str(target)), str)
            assert isinstance(llamacpp_ccache._link_parent(str(target)), str)
    finally:
        locked.chmod(0o600)
