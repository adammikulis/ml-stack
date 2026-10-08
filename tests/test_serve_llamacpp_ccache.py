"""ccache in the llama.cpp source build: the cmake flags and the environment it gets, the one extra
directory the sandbox lets it write (and that nothing else opens up), and the line the build says
when ccache is missing. None of this compiles llama.cpp."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from ml_stack import home
from ml_stack.sandbox import run
from ml_stack.sandbox.seatbelt import Seatbelt
from ml_stack.serve import llamacpp_ccache, llamacpp_compile
from ml_stack.serve.build_paths import BuildFailed
from ml_stack.serve.llamacpp_compile import Job, Toolchain

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
    assert "\n" not in line and line.startswith("ccache: 0 hits, 0 misses (n/a hit rate)")


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
