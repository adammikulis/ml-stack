"""The node launcher with real processes: start on demand, kill -9, checksum refusal, upgrade swap and rollback."""

from __future__ import annotations

import json
import os
import pwd
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest

from ml_stack import node_binary, node_launch, node_supervise, runtime
from ml_stack.node_health import call, node_health

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def built() -> Path:
    """The release node binary of this checkout, built once."""
    owner = Path(pwd.getpwuid(os.getuid()).pw_dir)  # the suite points HOME at a scratch directory; rustup needs the real one
    saved = {key: os.environ.get(key) for key in ("RUSTUP_HOME", "CARGO_HOME")}
    os.environ.setdefault("RUSTUP_HOME", str(owner / ".rustup"))
    os.environ.setdefault("CARGO_HOME", str(owner / ".cargo"))
    try:
        return node_binary.build(ROOT)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)


@pytest.fixture
def home(monkeypatch):
    """A short state root (a unix socket path is limited to about 100 bytes), with every node in it stopped afterwards."""
    root = Path(tempfile.mkdtemp(prefix="mln"))
    monkeypatch.setenv("ML_STACK_HOME", str(root))
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    yield root
    node_launch.stop_node(root / "node", wait_s=5)
    shutil.rmtree(root, ignore_errors=True)


def install_runtime(binary: Path, name: str, *, select: bool = True) -> Path:
    """A runtime tree holding ``binary`` as its node, selected the way `runtime.publish` writes it."""
    prefix = runtime.directory() / (name * 40)[:40] / uuid.uuid4().hex
    prefix.mkdir(parents=True)
    node_binary.install(prefix, binary, commit=name)
    if select:
        (runtime.directory() / "selected.json").write_text(json.dumps({"prefix": str(prefix)}), encoding="utf-8")
    return prefix


def clients(state: Path, count: int) -> list[subprocess.Popen]:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return [subprocess.Popen([sys.executable, "-m", "ml_stack.node_launch", "ensure", "--state", str(state)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(count)]


def processes(state: Path) -> list[str]:
    """The command lines of every process running a node or supervisor for this state directory."""
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if str(state) in line and ("poolside-node" in line or "node_launch" in line)
            and "ps -axo" not in line]


def board(state: Path) -> tuple[str, str]:
    """A session on the board `demo`: its name and token."""
    said = call(state, "register", {"model": "claude-sonnet-5-5", "harness": "claude-code", "session": "launcher-test"}, board="demo")
    return said["name"], said["token"]


def posts(state: Path, token: str) -> list[str]:
    read = call(state, "read", {"since": {}, "limit": 100}, board="demo", token=token)
    return [e["fields"]["body"] for e in read["entries"] if e["kind"] == "message"]


def post(state: Path, token: str, text: str) -> None:
    call(state, "post", {"kind": "message", "fields": {"type": "status", "body": text}}, board="demo", token=token)


def waited(what: str, ready, seconds: float = 15.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if (found := ready()):
            return found
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def test_eight_clients_start_exactly_one_node(home, built):
    install_runtime(built, "a")
    state = home / "node"
    runs = clients(state, 8)
    results = [run.communicate(timeout=60) for run in runs]
    assert [run.returncode for run in runs] == [0] * 8, results
    pids = {json.loads(out)["pid"] for out, _ in results}
    assert len(pids) == 1
    lines = processes(state)
    assert len([line for line in lines if "poolside-node run" in line]) == 1, lines
    assert len([line for line in lines if "node_launch supervise" in line]) == 1, lines
    health = node_launch.status(state)
    assert health["healthy"] and health["supervised"] and health["pid"] in pids and health["version"]
    assert health["uptime_s"] is not None and health["socket"].endswith("node.sock")


def test_kill_9_restarts_the_node_with_the_board_intact(home, built):
    install_runtime(built, "a")
    state = home / "node"
    node_launch.ensure_node(state)
    _, token = board(state)
    for index in range(5):
        post(state, token, f"row {index}")
    first = node_health(state)["pid"]
    os.kill(first, signal.SIGKILL)
    after = waited("a new node", lambda: (h := node_health(state)) and h["pid"] != first and h)
    assert after["pid"] != first
    again = call(state, "register", {"model": "claude-sonnet-5-5", "harness": "claude-code", "session": "launcher-test"}, board="demo")
    assert again["name"] and sorted(posts(state, again["token"])) == [f"row {i}" for i in range(5)]
    # ensure_node finds the restarted node and starts nothing more
    assert node_launch.ensure_node(state)["pid"] == after["pid"]
    assert len([line for line in processes(state) if "poolside-node run" in line]) == 1


def test_a_binary_that_does_not_match_its_checksum_is_refused(home, built):
    prefix = install_runtime(built, "a")
    with node_binary.location(prefix).open("ab") as stream:
        stream.write(b"tampered")
    state = home / "node"
    with pytest.raises(node_binary.NodeBinaryError, match="checksum"):
        node_launch.ensure_node(state)
    assert node_health(state) is None and processes(state) == []
    with pytest.raises(node_binary.NodeBinaryError, match="checksum"):
        node_binary.verified(prefix)


def test_swap_to_a_new_binary_keeps_the_board_and_the_socket(home, built):
    first = install_runtime(built, "a")
    state = home / "node"
    node_launch.ensure_node(state)
    _, token = board(state)
    post(state, token, "row before")
    old = node_launch.status(state)
    newer = home / "node-v2"
    shutil.copyfile(built, newer)
    with newer.open("ab") as stream:
        stream.write(b"\0" * 16)  # trailing bytes: a different checksum, the same program
    newer.chmod(0o700)
    install_runtime(newer, "b")
    done = node_launch.swap(state)
    assert done["action"] == "swapped", done
    now = node_launch.status(state)
    assert now["pid"] != old["pid"] and now["sha256"] != old["sha256"] and now["socket"] == old["socket"]
    assert "row before" in posts(state, board(state)[1])
    assert first.is_dir()
    assert len([line for line in processes(state) if "poolside-node run" in line]) == 1
    assert node_launch.swap(state)["action"] == "current"


def test_swap_to_an_unhealthy_binary_rolls_back_to_the_old_one(home, built):
    install_runtime(built, "a")
    state = home / "node"
    node_launch.ensure_node(state)
    _, token = board(state)
    post(state, token, "row before")
    old = node_launch.status(state)
    broken = home / "broken"
    broken.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    broken.chmod(0o700)
    install_runtime(broken, "c")
    done = node_launch.swap(state, wait_s=4)
    assert done["action"] == "failed" and "runs the previous binary again" in done["detail"], done
    now = node_launch.status(state)
    assert now["healthy"] and now["sha256"] == old["sha256"] and now["pinned"]
    assert "row before" in posts(state, board(state)[1])


def test_stop_ends_node_and_supervisor(home, built):
    install_runtime(built, "a")
    state = home / "node"
    node_launch.ensure_node(state)
    assert node_launch.stop_node(state)
    waited("the processes to go", lambda: processes(state) == [])
    assert node_supervise.running(state)["pid"]


def variant(home: Path, built: Path, name: str, tail: bytes) -> Path:
    """A copy of the node with other trailing bytes: another checksum, the same program."""
    copy = home / name
    shutil.copyfile(built, copy)
    with copy.open("ab") as stream:
        stream.write(tail)
    copy.chmod(0o700)
    return copy


def test_runtime_status_line_and_ensure_move_the_node(home, built):
    from ml_stack import runtime_cli, runtime_deploy

    state = home / "node"
    assert "NOT RUNNING" in runtime_cli.node_line()
    first = install_runtime(built, "a")
    node_launch.ensure_node(state)
    line = runtime_cli.node_line()
    assert "healthy" in line and "supervised" in line and str(state / "node.sock") in line and "pid" in line
    second = install_runtime(variant(home, built, "node-v3", b"\1"), "d")
    moved = runtime_cli._with_node(runtime_deploy.Outcome("switched", "d" * 40, "selected"))
    assert moved.action == "switched" and "node swapped" in moved.detail
    assert node_launch.status(state)["sha256"] == node_binary.record_of(second)["sha256"] != node_binary.record_of(first)["sha256"]
    broken = home / "broken"
    broken.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    broken.chmod(0o700)
    install_runtime(broken, "e")
    refused = runtime_cli._with_node(runtime_deploy.Outcome("switched", "e" * 40, "selected"))
    assert refused.action == "failed" and "previous binary again" in refused.detail
    assert "PINNED" in runtime_cli.node_line()


def test_packaging_builds_the_node_with_its_checksum(home, built, monkeypatch):
    owner = Path(pwd.getpwuid(os.getuid()).pw_dir)
    monkeypatch.setenv("RUSTUP_HOME", str(owner / ".rustup"))
    monkeypatch.setenv("CARGO_HOME", str(owner / ".cargo"))
    packaging = runpy.run_path(str(ROOT / "packaging" / "build.py"))
    made = packaging["node"](home / "dist")
    digest, name = (made.with_name(made.name + ".sha256").read_text(encoding="utf-8").split())
    assert name == made.name and digest == node_binary.sha256(made) and made.parent == home / "dist" / "node"


def test_a_runtime_whose_verification_mark_names_another_node_is_refused(home, built):
    prefix = install_runtime(built, "a", select=False)
    mark = prefix / "verified.json"
    mark.write_text(json.dumps({"node_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(node_binary.NodeBinaryError, match="checksum"):
        node_binary.verified(prefix)
    mark.write_text(json.dumps({"node_sha256": node_binary.record_of(prefix)["sha256"]}), encoding="utf-8")
    assert node_binary.verified(prefix).is_file()
    assert node_launch.smoke(prefix)["version"]
