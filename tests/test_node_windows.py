"""The node on Windows: the pipe name the two languages agree on, the stop events, and (on Windows only) the real node.

The first group runs anywhere with the Windows calls replaced by recorders; the last group builds the node and runs it
as a detached process under its supervisor, and is skipped everywhere else.
"""

from __future__ import annotations

import json
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest

from ml_stack import (
    features,
    lock,
    node_binary,
    node_health,
    node_launch,
    node_supervise,
    runtime,
    win32,
)
from ml_stack.platform import start_process

ROOT = Path(__file__).resolve().parent.parent


def test_the_pipe_key_is_the_one_the_rust_node_computes():
    # the same vector is asserted in app/poolside-node/src/sys/mod.rs
    want = "303c7027db4da393a0c6c375056106d2"
    assert node_health.key_of("C:\\Users\\Me\\State\\node") == want
    assert node_health.key_of("\\\\?\\c:\\users\\me\\state\\node\\") == want
    assert node_health.key_of("C:/Users/Me/State/node") == want


def test_every_state_directory_has_its_own_pipe_and_events(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert node_health.pipe_name(a) != node_health.pipe_name(b)
    assert node_health.pipe_name(a).startswith("\\\\.\\pipe\\poolside-node-")
    assert node_health.node_stop_event(a).startswith("Local\\poolside-node-stop-")
    assert node_supervise.supervisor_stop_event(a) != node_health.node_stop_event(a)


def reply_over_socketpair(reply: bytes, request: list):
    """The client end of a real duplex stream whose other end reads one request frame into ``request`` and answers ``reply``."""
    client, server = socket.socketpair()

    def serve():
        with server:
            (size,) = struct.unpack(">I", server.recv(4))
            request.append(json.loads(server.recv(size)))
            server.sendall(reply)
    threading.Thread(target=serve, daemon=True).start()
    return client.makefile("rwb", buffering=0)


def framed(value: dict) -> bytes:
    body = json.dumps(value).encode()
    return struct.pack(">I", len(body)) + body


def test_a_hello_over_a_pipe_is_framed_like_the_socket(monkeypatch, tmp_path):
    request, opened = [], []
    reply = framed({"ok": True, "result": {"node": "poolside-node", "pid": 7, "version": "0.2.2"}})
    monkeypatch.setattr(node_health, "is_windows", lambda: True)
    monkeypatch.setattr(win32, "open_pipe", lambda name: opened.append(name) or reply_over_socketpair(reply, request))
    said = node_health.node_health(tmp_path)
    assert said and said["pid"] == 7 and said["socket"] == node_health.pipe_name(tmp_path)
    assert opened == [node_health.pipe_name(tmp_path)]
    assert request == [{"v": 1, "id": 1, "method": "hello", "params": {}}]


def test_a_pipe_nobody_serves_is_a_dead_node(monkeypatch, tmp_path):
    def missing(name):
        raise FileNotFoundError(name)
    monkeypatch.setattr(node_health, "is_windows", lambda: True)
    monkeypatch.setattr(win32, "open_pipe", missing)
    assert node_health.node_health(tmp_path) is None


def test_a_busy_pipe_is_retried_until_it_opens(monkeypatch, tmp_path):
    tries = []

    def busy_then_open(name):
        tries.append(name)
        if len(tries) < 3:
            busy = OSError(0, "busy")
            busy.winerror = win32.ERROR_PIPE_BUSY  # type: ignore[attr-defined]  # set by the OS on Windows only
            raise busy
        return reply_over_socketpair(framed({"ok": True, "result": {"pid": 1}}), [])
    monkeypatch.setattr(node_health, "is_windows", lambda: True)
    monkeypatch.setattr(win32, "open_pipe", busy_then_open)
    assert node_health.node_health(tmp_path)["pid"] == 1 and len(tries) == 3


def test_the_supervisor_stops_on_its_event_and_on_ctrl_break(monkeypatch, tmp_path):
    events, closed, handlers = {}, [], {}

    def event_named(name):
        events[name] = threading.Event()
        events[name].close = lambda: closed.append(name)
        return events[name]
    monkeypatch.setattr(node_supervise, "is_windows", lambda: True)
    monkeypatch.setattr(win32, "Event", event_named)
    monkeypatch.setattr(signal, "SIGBREAK", 21, raising=False)
    monkeypatch.setattr(signal, "signal", lambda number, handler: handlers.update({number: handler}))
    stopping, close = node_supervise._stop_requests(tmp_path)
    event = events[node_supervise.supervisor_stop_event(tmp_path)]
    assert not stopping()
    event.set()
    assert stopping()
    event.clear()
    handlers[21](21, None)
    assert stopping(), "Ctrl+Break asks it to stop"
    close()
    assert closed == [node_supervise.supervisor_stop_event(tmp_path)]


def test_a_stop_on_windows_signals_the_supervisor_and_the_node_by_event(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(node_launch, "is_windows", lambda: True)
    monkeypatch.setattr(win32, "signal_event", lambda name: sent.append(name) or True)
    monkeypatch.setattr(node_launch, "node_health", lambda state: None)
    assert node_launch.stop_node(tmp_path, wait_s=2)
    assert sent == [node_supervise.supervisor_stop_event(tmp_path), node_health.node_stop_event(tmp_path)]


def test_a_process_probe_on_windows_never_sends_a_signal(monkeypatch):
    sent = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(win32, "process_alive", lambda pid: sent.append(pid) or True)
    monkeypatch.setattr("os.kill", lambda *a: pytest.fail("os.kill(pid, 0) is CTRL_C_EVENT on Windows"))
    assert lock.pid_alive(1234) and sent == [1234]


# -- the real node ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def built() -> Path:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ML_STACK_HOME", tempfile.mkdtemp(prefix="mlf"))
        return node_binary.build(ROOT)


@pytest.fixture
def home(monkeypatch):
    root = Path(tempfile.mkdtemp(prefix="mln"))
    monkeypatch.setenv("ML_STACK_HOME", str(root))
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    yield root
    node_launch.stop_node(root / "node", wait_s=5)


def select(binary: Path) -> Path:
    prefix = runtime.directory() / (uuid.uuid4().hex)[:40] / uuid.uuid4().hex
    prefix.mkdir(parents=True)
    node_binary.install(prefix, binary, commit="windows")
    (runtime.directory() / "selected.json").write_text(json.dumps({"prefix": str(prefix)}), encoding="utf-8")
    return prefix


@pytest.mark.skipif(sys.platform != "win32", reason="needs the Windows node")
def test_the_node_binary_is_an_exe(built):
    assert built.name == "poolside-node.exe"


@pytest.mark.skipif(sys.platform != "win32", reason="needs the Windows node")
def test_the_node_answers_over_its_pipe_and_leaves_on_its_stop_event(built, home):
    state = home / "n"
    child = start_process([str(built), "run", "--state", str(state)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 20
        said = None
        while said is None and time.monotonic() < deadline:
            said = node_health.node_health(state)
            time.sleep(0.05)
        assert said and said["pid"] == child.pid and said["socket"] == node_health.pipe_name(state)
        assert win32.process_alive(child.pid)
        assert win32.signal_event(node_health.node_stop_event(state))
        assert child.wait(timeout=10) == 0, "a graceful stop exits cleanly"
        assert node_health.node_health(state) is None and not win32.process_alive(child.pid)
    finally:
        if child.poll() is None:
            child.kill()


@pytest.mark.skipif(sys.platform != "win32", reason="needs the Windows node")
def test_ensure_starts_the_node_under_its_supervisor_and_stop_ends_both(built, home):
    select(built)
    state = home / "node"
    said = node_launch.ensure_node(state)
    assert said["pid"] and node_launch.supervised(state)
    assert node_launch.status(state)["socket"] == node_health.pipe_name(state)
    assert node_launch.stop_node(state, wait_s=15)
    assert not node_launch.supervised(state)
