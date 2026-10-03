"""Following upstream llama.cpp: a real git repository on loopback stands in for GitHub, its commits
build a stub ``llama-server`` with real cmake in the sandbox, and the build is pinned, smoke-tested and
switched to (or refused) for real. No llama.cpp is compiled."""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

import ml_stack.setup as stack_setup
from ml_stack import sentinel
from ml_stack.serve import (
    binary,
    llamacpp_cli,
    llamacpp_compile,
    llamacpp_smoke,
    llamacpp_state,
    llamacpp_status,
    llamacpp_update,
)
from ml_stack.serve.build_paths import BuildFailed, builds_dir, current_link, root
from ml_stack.serve.llamacpp_smoke import Result
from tests.llamacpp_site import UpstreamSite, toolchain


def passes(binary_path, model=None, say=None):
    return Result([("health", True, "ok")])


def refuses(binary_path, model=None, say=None):
    return Result([("health", True, "ok"), ("top_logprobs", False, "top_logprobs came back empty")])


@pytest.fixture
def site(tmp_path, monkeypatch):
    if shutil.which("cmake") is None or shutil.which("git") is None:
        pytest.skip("needs cmake and git")
    monkeypatch.setenv("GIT_SSL_NO_VERIFY", "1")
    made = UpstreamSite(tmp_path / "up")
    yield made
    made.close()


@pytest.fixture
def pipe(site, tmp_path):
    return site.pipeline(tmp_path)


def run(site, pipe, ref="master", smoke=passes, **more):
    env = llamacpp_update.Env(site.upstream, pipe, toolchain(), 2, smoke)
    return llamacpp_update.update(ref, env, **more)


def cli(site, pipe, *argv):
    args = argparse.Namespace(action=argv[0], build="", ref="master", jobs=2, force=False, model="", off=False,
                              keep=3, offline=False, yes=False, json=False)
    rest = list(argv[1:])
    while rest:
        key = rest.pop(0)
        if key == "--build":
            args.build = rest.pop(0)
        elif key == "--ref":
            args.ref = rest.pop(0)
        else:
            setattr(args, key.removeprefix("--"), True)
    return llamacpp_cli.cmd_llama_cpp(args, upstream=site.upstream, pipeline=pipe)


def test_an_update_builds_the_commit_pins_the_binary_and_makes_it_active(site, pipe):
    sha = site.commit(100)
    out = run(site, pipe)
    assert (out.status, out.build) == ("built", f"b100-{sha[:9]}")
    dest = builds_dir() / out.build
    info = json.loads((dest / "BUILD.json").read_text())
    assert info["commit"] == sha and info["build"] == 100 and info["track"] is True
    assert info["version"] == "version: 100 (build 100, commit stub)"
    assert info["smoke"]["passed"] is True
    pin = sentinel.default().manifest.pins()[str((dest / "llama-server").resolve())]
    assert (pin.kind, pin.source, pin.sha256) == ("binary", "build", info["sha256"])
    assert binary.find_binary("llama-server") == (dest / "llama-server").resolve()
    assert current_link().resolve() == dest.resolve()
    assert not (root() / "work").exists() or not list((root() / "work").iterdir())


def test_the_checkout_is_the_asked_commit_and_carries_no_repository(site, pipe, tmp_path):
    first = site.commit(100)
    site.commit(101)
    where = tmp_path / "checkout"
    llamacpp_compile.checkout(site.upstream, first, where, pipe)
    assert (where / "NOTE").read_text() == "100" and not (where / ".git").exists()


def test_a_ref_that_is_a_tag_builds_that_commit(site, pipe):
    old = site.commit(100)
    site.commit(101)
    out = run(site, pipe, "b100")
    assert out.build == f"b100-{old[:9]}"


def test_a_repository_template_hook_does_not_run_during_the_fetch(site, pipe, tmp_path, monkeypatch):
    template = tmp_path / "template" / "hooks"
    template.mkdir(parents=True)
    marker = tmp_path / "hook-ran"
    for name in ("post-checkout", "post-merge", "reference-transaction"):
        (template / name).write_text(f"#!/bin/sh\ntouch {marker}\n")
        (template / name).chmod(0o755)
    monkeypatch.setenv("GIT_TEMPLATE_DIR", str(tmp_path / "template"))
    site.commit(100)
    assert run(site, pipe).status == "built"
    assert not marker.exists()


def test_a_second_update_keeps_the_first_for_rollback(site, pipe):
    first = site.commit(100)
    site.commit(101)
    one = run(site, pipe, f"{first}")
    two = run(site, pipe)
    assert llamacpp_state.active().name == two.build
    assert llamacpp_state.load()["previous"] == [one.build]
    assert llamacpp_state.rollback().name == one.build
    assert binary.find_binary("llama-server").parent.name == one.build
    with pytest.raises(LookupError, match="no earlier build"):
        llamacpp_state.rollback()


def test_a_build_that_fails_its_smoke_test_leaves_the_old_build_active_and_is_kept(site, pipe, capsys):
    site.commit(100)
    good = run(site, pipe)
    bad_sha = site.commit(101)
    out = run(site, pipe, smoke=refuses)
    assert out.status == "failed" and "top_logprobs" in out.detail
    assert llamacpp_state.active().name == good.build
    assert not (builds_dir() / out.build).exists()
    kept = out.kept
    assert kept.parent == root() / "failed" and (kept / "llama-server").is_file()
    info = json.loads((kept / "BUILD.json").read_text())
    assert info["commit"] == bad_sha and info["smoke"]["passed"] is False
    assert str(kept / "llama-server") not in sentinel.default().manifest.pins()
    assert binary.find_binary("llama-server").parent.name == good.build


def test_a_tampered_managed_binary_is_refused_and_quarantined(site, pipe):
    site.commit(100)
    out = run(site, pipe)
    target = builds_dir() / out.build / "llama-server"
    with target.open("ab") as handle:
        handle.write(b"# patched\n")
    with pytest.raises(binary.BinaryTampered, match="pin"):
        binary.find_binary("llama-server")
    assert sentinel.default().store.blocked("binary", str(target.resolve()))
    with pytest.raises(binary.BinaryTampered, match="quarantined"):
        binary.find_binary("llama-server")


def test_a_binary_swapped_for_another_one_with_its_pin_removed_is_still_refused(site, pipe):
    site.commit(100)
    out = run(site, pipe)
    target = builds_dir() / out.build / "llama-server"
    sentinel.default().manifest.unpin(target)
    target.write_text("#!/bin/sh\necho evil\n")
    with pytest.raises(binary.BinaryTampered, match="not the binary that was built"):
        binary.find_binary("llama-server")


def test_discovery_prefers_an_env_override_then_the_managed_build_then_the_machine(site, pipe, tmp_path, monkeypatch):
    site.commit(100)
    out = run(site, pipe)
    managed = (builds_dir() / out.build / "llama-server").resolve()
    override = tmp_path / "override" / "llama-server"
    override.parent.mkdir()
    override.write_text("#!/bin/sh\n")
    monkeypatch.setenv("LLAMA_CPP_SERVER", str(override))
    assert binary.find_binary("llama-server") == override.resolve()
    monkeypatch.delenv("LLAMA_CPP_SERVER")
    assert binary.find_binary("llama-server") == managed
    current_link().unlink()
    elsewhere = binary.find_binary("llama-server")
    assert elsewhere != managed and (elsewhere is None or root() not in elsewhere.parents)


def test_pin_stops_tracking_and_update_refuses_until_unpinned(site, pipe):
    site.commit(100)
    one = run(site, pipe)
    site.commit(101)
    two = run(site, pipe)
    assert cli(site, pipe, "pin", "--build", one.build) == 0
    assert llamacpp_state.active().name == one.build and llamacpp_state.load()["pinned"] == one.build
    with pytest.raises(BuildFailed, match="pinned"):
        run(site, pipe)
    assert cli(site, pipe, "update") == 2
    assert cli(site, pipe, "pin", "--off") == 0
    assert run(site, pipe).build == two.build
    assert cli(site, pipe, "pin", "--build", "b999-nothing") == 2


def test_prune_keeps_three_good_builds_and_asks_before_deleting(site, pipe, monkeypatch, capsys):
    names = []
    for number in range(100, 106):
        site.commit(number)
        names.append(run(site, pipe).build)
    assert [b.name for b in llamacpp_state.prune_candidates()] == names[:3]
    monkeypatch.setattr("builtins.input", lambda _prompt="": "no")
    assert cli(site, pipe, "prune") == 0
    assert all((builds_dir() / n).is_dir() for n in names)
    monkeypatch.setattr("builtins.input", lambda _prompt="": pytest.fail("--yes asks nothing"))
    assert cli(site, pipe, "prune", "--yes") == 0
    assert [n for n in names if (builds_dir() / n).is_dir()] == names[3:]
    assert str((builds_dir() / names[0] / "llama-server").resolve()) not in sentinel.default().manifest.pins()
    assert llamacpp_state.active().name == names[-1]


def test_a_host_that_needs_approval_ends_in_the_needs_approval_state_with_the_exact_line(site, tmp_path, capsys):
    site.commit(100)
    refusing = site.pipeline(tmp_path, allowed=("github.com",))
    assert cli(site, refusing, "update") == llamacpp_cli.NEEDS_APPROVAL
    assert "ml-stack-security approve-host 127.0.0.1" in capsys.readouterr().err
    assert not builds_dir().exists() or not list(builds_dir().iterdir())
    assert llamacpp_status.gather(site.upstream, refusing)["newer"] == "unknown (source host not approved)"


def test_status_says_where_the_binary_came_from_and_whether_upstream_is_ahead(site, pipe, capsys):
    site.commit(100)
    site.stable = "v0.5.0"
    out = run(site, pipe)
    got = llamacpp_status.gather(site.upstream, pipe)
    assert got["origin"] == "managed build (track)" and got["build"] == 100 and got["newer"] == "no"
    assert got["upstream"]["tag"] == "b100" and got["upstream"]["stable"] == "v0.5.0"
    site.commit(101)
    assert llamacpp_status.gather(site.upstream, pipe)["newer"] == "yes"
    assert cli(site, pipe, "status") == 0
    text = capsys.readouterr().out
    assert out.build.split("-")[0] in text or "managed build (track)" in text
    assert "newer upstream commit: yes" in text


def test_status_offline_says_unknown(site, pipe):
    site.commit(100)
    run(site, pipe)
    site.close()
    got = llamacpp_status.gather(site.upstream, pipe)
    assert got["newer"] == "unknown (offline)" and got["upstream_note"]
    assert llamacpp_status.gather(site.upstream, pipe, check=False)["upstream"] == {}


def test_a_missing_tool_is_named_and_nothing_is_installed(site, pipe, monkeypatch):
    site.commit(100)
    asked = []
    real = subprocess.run
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None if name == "cmake" else real and "/usr/bin/" + name)
    monkeypatch.setattr(subprocess, "run", lambda argv, *a, **k: asked.append(argv) or real(argv, *a, **k))
    with pytest.raises(BuildFailed) as caught:
        llamacpp_update.update("master", llamacpp_update.Env(site.upstream, pipe, smoke=passes))
    assert "missing cmake" in str(caught.value) and "installs nothing" in str(caught.value)
    assert not any("brew" in str(a) or "install" in str(a) for a in asked)
    assert site.hits == []


def test_a_ref_that_is_not_a_ref_is_refused_before_any_git_command(site, pipe):
    site.commit(100)
    for bad in ("--upload-pack=touch x", "a b", "../x", "x..y", ""):
        with pytest.raises(BuildFailed):
            run(site, pipe, bad)
    assert not any("git-upload-pack" in hit for hit in site.hits)


def test_the_compile_cannot_write_outside_its_directory(site, pipe, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "written"
    site.commit(100)
    cmake = site.work / "CMakeLists.txt"
    cmake.write_text(cmake.read_text() + f"\nadd_custom_command(TARGET llama-server PRE_BUILD COMMAND ${{CMAKE_COMMAND}} -E touch {marker})\n")
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "escape"], cwd=site.work, check=True)
    subprocess.run(["git", "push", "-q", "origin", "master"], cwd=site.work, check=True)
    with pytest.raises(BuildFailed, match="compile failed"):
        run(site, pipe)
    assert not marker.exists()


def test_the_compile_has_no_network(site, pipe, tmp_path):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(5)
    listener.settimeout(0.5)
    site.commit(100)
    cmake = site.work / "CMakeLists.txt"
    cmake.write_text(cmake.read_text() + f"\nfile(DOWNLOAD http://127.0.0.1:{listener.getsockname()[1]}/x "
                     "${CMAKE_BINARY_DIR}/got TIMEOUT 5)\n")
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "net"], cwd=site.work, check=True)
    subprocess.run(["git", "push", "-q", "origin", "master"], cwd=site.work, check=True)
    assert run(site, pipe).status == "built"
    reached_by_the_compile = False
    try:
        while True:                                  # other processes on a busy machine may probe a loopback port;
            conn, _ = listener.accept()              # only the compile's own request (GET /x) counts
            conn.settimeout(0.5)
            try:
                reached_by_the_compile |= b"GET /x" in conn.recv(2048)
            except OSError:
                pass
            finally:
                conn.close()
    except TimeoutError:
        pass
    finally:
        listener.close()
    assert not reached_by_the_compile


def test_the_smoke_test_without_a_model_or_a_working_server_fails_instead_of_passing(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_SMOKE_GGUF", str(tmp_path / "missing.gguf"))
    script = tmp_path / "llama-server"
    script.write_text("#!/bin/sh\nexit 1\n")
    script.chmod(0o755)
    got = llamacpp_smoke.run(script, tmp_path / "missing.gguf", timeout=20)
    assert not got.passed and got.checks[0][0] in ("lease", "model")
    monkeypatch.setattr(llamacpp_smoke, "smallest_model", lambda: None)
    assert not llamacpp_smoke.run(script).passed


def test_origin_is_named_for_each_place_a_binary_can_come_from(tmp_path, monkeypatch):
    assert llamacpp_status.origin_of(Path("/opt/homebrew/Cellar/llama.cpp/0.5.0/bin/llama-server")) == "homebrew"
    assert llamacpp_status.origin_of(Path("/usr/local/bin/llama-server")) == "PATH"
    monkeypatch.setenv("LLAMA_CPP_SERVER", str(tmp_path / "x" / "llama-server"))
    assert llamacpp_status.origin_of(tmp_path / "x" / "llama-server").startswith("env override")


def test_the_architectures_of_a_statically_linked_server_are_read_from_the_binary(tmp_path):
    server = tmp_path / "llama-server"
    server.write_bytes(b"\x00\x01qwen4exp\x00gemma4\x00unrelated\x00")
    assert stack_setup._arches(server, known={"qwen4exp", "gemma4", "bert"}) == {"qwen4exp", "gemma4"}
