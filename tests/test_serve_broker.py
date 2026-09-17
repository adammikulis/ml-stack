"""The broker: servers are asked for by purpose, shared, queued and reaped -- never taken
from a process that still holds them.

Every server here is a real process bound to a real port (`fake_llama_binary`), and every
holder is a real process whose pid can end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from ml_stack.serve import LlamaServerBackend, ServerManager
from ml_stack.serve.broker import Ask, Broker, BrokerError
from ml_stack.serve.process import every_server, kill_process_tree, pid_exists
from ml_stack.testing.fakes import fake_llama_binary


@pytest.fixture
def llama_binary(tmp_path, monkeypatch):
    from ml_stack.serve import backend as backend_module

    monkeypatch.setattr(backend_module, "log_dir", lambda: tmp_path / "logs")
    return fake_llama_binary(tmp_path)


@pytest.fixture
def broker(tmp_path, llama_binary):
    manager = ServerManager(LlamaServerBackend(binary=llama_binary),
                            state_file=tmp_path / "servers.json")
    made = Broker(manager, idle_s=3600.0, room=lambda: None, scan=lambda: [])
    yield made
    for held in list(made.servers.values()):
        if held.pid and held.ours:
            kill_process_tree(held.pid)


@pytest.fixture
def models(tmp_path):
    out = []
    for name in ("first-M1.gguf", "second-M2.gguf"):
        path = tmp_path / name
        path.write_bytes(b"GGUF" + b"\x00" * 64)
        out.append(str(path))
    return out


@pytest.fixture
def holders():
    """Processes that stand in for clients: each one's pid holds a lease until it ends."""
    started: list[subprocess.Popen] = []

    def one() -> subprocess.Popen:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        started.append(proc)
        return proc

    yield one
    for proc in started:
        proc.kill()
        proc.wait(timeout=10)


def ask(model: str, pid: int, purpose: str = "chat") -> Ask:
    return Ask(purpose=purpose, models=(model,), pid=pid, label=f"pid {pid}",
               spec={"context": 512})


def test_two_asks_for_one_model_share_one_server(broker, models, holders):
    a, b = holders(), holders()
    first = broker.lease(ask(models[0], a.pid), timeout=30)
    second = broker.lease(ask(models[0], b.pid), timeout=30)

    assert second.port == first.port
    assert second.shared and not first.shared
    held = broker.servers[first.port]
    assert {pid for pid, _ in held.holders.values()} == {a.pid, b.pid}
    assert len(broker.servers) == 1


def test_a_conflicting_model_waits_for_the_holder_and_kills_nothing_it_holds(
        broker, models, holders):
    a, b = holders(), holders()
    first = broker.lease(ask(models[0], a.pid), timeout=30)
    first_pid = broker.servers[first.port].pid
    got: dict = {}

    def second() -> None:
        got["grant"] = broker.lease(ask(models[1], b.pid), timeout=30)

    waiter = threading.Thread(target=second)
    waiter.start()
    time.sleep(2.0)

    queue = broker.snapshot()["queue"]
    assert [w["pid"] for w in queue] == [b.pid]
    assert f"pid {a.pid}" in queue[0]["blocked_by"]
    assert pid_exists(first_pid), "the held server was stopped for a conflicting ask"
    assert "grant" not in got

    broker.release(first.lease)
    waiter.join(timeout=30)
    assert got["grant"].model == models[1]
    assert not broker.snapshot()["queue"]


def test_a_dead_holders_lease_is_reaped_and_the_queue_advances(broker, models, holders):
    a, b = holders(), holders()
    broker.lease(ask(models[0], a.pid), timeout=30)
    got: dict = {}
    waiter = threading.Thread(
        target=lambda: got.setdefault("grant", broker.lease(ask(models[1], b.pid), timeout=30)))
    waiter.start()
    time.sleep(1.0)
    assert "grant" not in got

    a.kill()
    a.wait(timeout=10)
    broker.reap()
    waiter.join(timeout=30)
    assert got["grant"].model == models[1]


def test_a_timed_out_ask_says_who_holds_what_it_wanted(broker, models, holders):
    a, b = holders(), holders()
    broker.lease(ask(models[0], a.pid), timeout=30)
    with pytest.raises(BrokerError, match=f"held by pid {a.pid}"):
        broker.lease(ask(models[1], b.pid), timeout=1.0)


def test_a_holder_switching_its_own_model_is_not_blocked_by_itself(broker, models, holders):
    a = holders()
    broker.lease(ask(models[0], a.pid), timeout=30)
    switched = broker.lease(ask(models[1], a.pid), timeout=5)
    assert switched.model == models[1]
    assert [h.model for h in broker.servers.values()] == [models[1]]


def test_a_held_server_is_not_stopped_on_request_and_a_foreign_one_never(
        broker, models, holders, llama_binary, tmp_path):
    a = holders()
    grant = broker.lease(ask(models[0], a.pid), timeout=30)
    assert broker.stop(grant.port)["stopped"] is False

    from ml_stack.serve.ports import free_port

    port = free_port()
    foreign = subprocess.Popen([str(llama_binary), "--port", str(port), "-m", models[1]])
    try:
        deadline = time.monotonic() + 20
        broker.scan = lambda: [s for s in every_server() if s["pid"] == foreign.pid]
        while time.monotonic() < deadline and port not in broker.servers:
            broker.adopt()
            time.sleep(0.2)
        assert port in broker.servers and broker.servers[port].ours is False
        assert broker.stop(port)["stopped"] is False
        shared = broker.lease(ask(models[1], a.pid, purpose="embed"), timeout=10)
        assert shared.port == port and shared.shared
        assert foreign.poll() is None
    finally:
        foreign.kill()
        foreign.wait(timeout=10)


@pytest.mark.slow
def test_processes_lease_through_the_broker_they_start(tmp_path, llama_binary, models):
    """Real entry point: each client process calls `broker_wire.lease`, which starts the
    machine's broker when none answers. A conflicting ask waits until its holder ends."""
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "LLAMA_CPP_SERVER": str(llama_binary),
           "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH")) if p)}
    client = ("import json, sys, time\n"
              "from ml_stack.serve import broker_wire\n"
              "g = broker_wire.lease('chat', [sys.argv[1]], spec={'context': 512}, timeout=60)\n"
              "print(json.dumps(g.as_dict()), flush=True)\n"
              "time.sleep(float(sys.argv[2]))\n")

    def run(model: str, hold: float) -> subprocess.Popen:
        return subprocess.Popen([sys.executable, "-c", client, model, str(hold)], env=env,
                                stdout=subprocess.PIPE, text=True)

    from ml_stack.serve import broker_wire

    holder = run(models[0], 60)
    sharer = run(models[0], 1)
    try:
        first = json.loads(holder.stdout.readline())
        shared = json.loads(sharer.stdout.readline())
        assert shared["port"] == first["port"]
        assert {first["shared"], shared["shared"]} == {True, False}

        waiter = run(models[1], 1)
        time.sleep(3.0)
        snapshot = broker_wire.status()
        assert [w["pid"] for w in snapshot["queue"]] == [waiter.pid]
        assert pid_exists(next(s["pid"] for s in snapshot["servers"]
                               if s["port"] == first["port"]))

        holder.kill()
        holder.wait(timeout=10)
        granted = json.loads(waiter.stdout.readline())
        assert granted["model"] == models[1]
        assert waiter.wait(timeout=30) == 0
    finally:
        for proc in (holder, sharer):
            proc.kill()
            proc.wait(timeout=10)
        record = json.loads(broker_wire.record_path().read_text())
        for server in broker_wire.status()["servers"]:
            if server["pid"]:
                kill_process_tree(server["pid"])
        kill_process_tree(record["pid"])
