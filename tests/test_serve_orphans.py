"""Servers a dead host left behind are stopped by the next start, and only those."""

import json
import subprocess
import sys
import time

import psutil
import pytest

from ml_stack.serve import LlamaServerBackend, ServerManager, ServerSpec, free_port
from ml_stack.serve.leases import orphaned, same_process
from ml_stack.serve.process import cmdline_digest, pid_exists, started_at
from ml_stack.testing import fake_llama_binary


def _sleeper(*extra):
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", *extra])


def _dead_pid():
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    return done.pid


def _record(proc, port, *, owner, started=True, cmdline=True):
    entry = {"port": port, "pid": proc.pid, "owner_pid": owner, "model": "m.gguf",
             "backend": "llama-server"}
    if started:
        entry["started"] = started_at(proc.pid)
    if cmdline:
        entry["cmdline"] = cmdline_digest(proc.pid)
    return entry


@pytest.fixture
def children():
    made = []
    yield made
    for proc in made:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def test_a_record_proves_its_process_by_start_time_and_command_line(children):
    proc = _sleeper()
    children.append(proc)
    good = _record(proc, 9001, owner=_dead_pid())
    assert same_process(good) and same_process(good, strict=True)
    assert not same_process({**good, "started": good["started"] + 60})
    assert not same_process({**good, "cmdline": "0" * 64}), "same pid, another program"


def test_a_record_with_nothing_to_prove_it_is_trusted_loosely_and_refused_strictly(children):
    proc = _sleeper()
    children.append(proc)
    bare = _record(proc, 9001, owner=_dead_pid(), started=False, cmdline=False)
    assert same_process(bare) and not same_process(bare, strict=True)
    half = _record(proc, 9001, owner=_dead_pid(), cmdline=False)
    assert same_process(half) and not same_process(half, strict=True)


def test_an_orphan_is_a_server_whose_owner_has_gone(children):
    proc = _sleeper()
    children.append(proc)
    assert orphaned(_record(proc, 9001, owner=_dead_pid()), strict=True)
    assert not orphaned(_record(proc, 9001, owner=proc.pid))
    assert not orphaned(_record(proc, 9001, owner=psutil.Process().pid))


def _lease_elsewhere(tmp_path, state, **kw):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 64)
    manager = ServerManager(backend=LlamaServerBackend(binary=fake_llama_binary(tmp_path)),
                            state_file=state, **kw)
    info = manager.lease(ServerSpec(model=model, port=free_port()), roam=False, timeout=30.0,
                         check_flags=False, preflight=False, warmup_request=False)
    return manager, info


@pytest.mark.slow
def test_the_next_start_stops_a_verified_orphan_on_another_port(tmp_path, children):
    orphan, stranger = _sleeper(), _sleeper("unrelated")
    children.extend([orphan, stranger])
    state = tmp_path / "servers.json"
    state.write_text(json.dumps({
        "9001": _record(orphan, 9001, owner=_dead_pid()),
        "9002": _record(stranger, 9002, owner=_dead_pid(), cmdline=False),
        "9003": {**_record(stranger, 9003, owner=_dead_pid()), "cmdline": "0" * 64},
    }))
    told = []
    manager = ServerManager(backend=LlamaServerBackend(binary=fake_llama_binary(tmp_path)),
                            state_file=state)
    manager.say = told.append
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 64)
    info = manager.lease(ServerSpec(model=model, port=free_port()), roam=False, timeout=30.0,
                         check_flags=False, preflight=False, warmup_request=False)
    try:
        deadline = time.monotonic() + 10
        while orphan.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert orphan.poll() is not None, "verified, so stopped"
        assert stranger.poll() is None, "its record could not prove it, so it was left alone"
        assert any("orphaned server" in line and "9001" in line for line in told)
        assert pid_exists(info.pid)
    finally:
        manager.release(info)


@pytest.mark.slow
def test_a_server_whose_owner_is_alive_is_not_swept(tmp_path, children):
    live = _sleeper()
    children.append(live)
    state = tmp_path / "servers.json"
    state.write_text(json.dumps({"9001": _record(live, 9001, owner=psutil.Process().pid)}))
    manager, info = _lease_elsewhere(tmp_path, state)
    try:
        assert live.poll() is None
    finally:
        manager.release(info)
