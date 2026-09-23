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
def llama_binary(tmp_path):
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
    tag = tmp_path.name  # a model name no concurrently running test serves
    for name in (f"first-M1-{tag}.gguf", f"second-M2-{tag}.gguf"):
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


def test_test_runners_share_the_machine_s_cores(broker, holders):
    """A second suite gets what the first left, and never nothing."""
    broker.cpus = 8
    first, second = holders(), holders()
    mine = broker.take_cores(first.pid, 8)
    assert mine["cores"] == 8
    theirs = broker.take_cores(second.pid, 8)
    assert theirs["cores"] == 1, "a suite beside a full one must run narrow, not refuse"

    broker.give_back_cores(mine["lease"])
    assert broker.take_cores(second.pid, 4)["cores"] == 4
    assert broker.snapshot()["cores"]["cpus"] == 8


def test_a_dead_runners_cores_go_back(broker, holders):
    broker.cpus = 4
    gone = holders()
    assert broker.take_cores(gone.pid, 4)["cores"] == 4
    gone.kill()
    gone.wait(timeout=10)
    broker.reap()
    assert broker.take_cores(holders().pid, 4)["cores"] == 4


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


def test_a_server_started_outside_the_broker_is_shared_not_loaded_twice(
        broker, models, holders, llama_binary):
    from ml_stack.client import is_healthy
    from ml_stack.serve.ports import free_port

    port = free_port()
    foreign = subprocess.Popen([str(llama_binary), "--port", str(port), "-m", models[0]])
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not is_healthy(f"http://127.0.0.1:{port}"):
            time.sleep(0.2)
        broker.scan = lambda: [s for s in every_server() if s["pid"] == foreign.pid]
        grant = broker.lease(ask(models[0], holders().pid), timeout=10)
        assert grant.port == port and grant.shared
        assert list(broker.servers) == [port]
    finally:
        foreign.kill()
        foreign.wait(timeout=10)


def test_a_restarted_broker_keeps_a_live_holders_lease(broker, models, holders, llama_binary, tmp_path):
    """A broker starting where another left off must not unload a server whose holder is
    still using it."""
    a = holders()
    first = broker.lease(ask(models[0], a.pid), timeout=30)

    successor = Broker(ServerManager(LlamaServerBackend(binary=llama_binary),
                                     state_file=tmp_path / "servers.json"),
                       idle_s=0.0, room=lambda: None, scan=lambda: [])
    successor.adopt()
    held = successor.servers[first.port]
    assert {pid for pid, _ in held.holders.values()} == {a.pid}
    assert held.purpose == "chat"
    assert not successor.reap(), "a held server was unloaded as idle after the restart"
    assert pid_exists(held.pid)


def test_a_server_put_up_by_hand_is_held_by_itself_and_never_reaped(
        models, holders, llama_binary, tmp_path):
    """`ml-stack-serve up` records the server as its own owner; the broker never stops it."""
    from ml_stack.serve import ServerSpec
    from ml_stack.serve.ports import free_port

    manager = ServerManager(LlamaServerBackend(binary=llama_binary),
                            state_file=tmp_path / "servers.json")
    info = manager.lease(ServerSpec(model=models[0], port=free_port(), context=512),
                         roam=False, timeout=30.0, check_flags=False, preflight=False,
                         warmup_request=False)
    manager.detach(info)
    said: list[str] = []
    watcher = Broker(ServerManager(LlamaServerBackend(binary=llama_binary),
                                   state_file=tmp_path / "servers.json"),
                     idle_s=0.0, room=lambda: 0, scan=lambda: [])
    watcher.say = said.append
    try:
        watcher.adopt()
        held = watcher.servers[info.port]
        assert [label for _, label in held.holders.values()] == ["ml-stack-serve up"]
        time.sleep(0.05)
        assert watcher.reap() == [] and pid_exists(info.pid)
        with pytest.raises(BrokerError):
            watcher.lease(ask(models[1], holders().pid), timeout=1.0)
        assert pid_exists(info.pid), "evicted to make room"
        assert said == []
    finally:
        kill_process_tree(info.pid)


def test_a_server_nobody_holds_is_not_reaped_while_it_is_answering(broker, models, holders):
    a = holders()
    grant = broker.lease(ask(models[0], a.pid), timeout=30)
    broker.release(grant.lease)
    broker.idle_s = 0.0
    said: list[str] = []
    broker.say = said.append
    broker.busy = lambda url: True
    time.sleep(0.05)
    assert broker.reap() == [] and grant.port in broker.servers
    broker.busy = lambda url: False
    time.sleep(0.05)
    assert [h.port for h in broker.reap()] == [grant.port]
    assert said and f"stopping port {grant.port}" in said[0] and "answering nothing" in said[0]


def test_a_lease_waits_out_a_measurement_instead_of_failing(broker, models, holders):
    """The card being measured is a holder like any other: the lease waits for it."""
    from ml_stack.serve.manager import Measuring

    started = broker.manager.lease
    refusals = {"left": 2}

    def measuring_first(spec, **kwargs):
        if refusals["left"]:
            refusals["left"] -= 1
            raise Measuring("the card is being measured by ml-stack-bench (pid 1)")
        return started(spec, **kwargs)

    broker.manager.lease = measuring_first
    grant = broker.lease(ask(models[0], holders().pid), timeout=30)
    assert refusals["left"] == 0
    assert grant.model == models[0] and not grant.shared


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
    sharer = run(models[0], 10)
    try:
        first = json.loads(holder.stdout.readline())
        shared = json.loads(sharer.stdout.readline())
        assert shared["port"] == first["port"]
        held = next(s for s in broker_wire.status()["servers"] if s["port"] == first["port"])
        assert {h["pid"] for h in held["holders"]} == {holder.pid, sharer.pid}

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


def _broker_process(tmp_path: Path, *extra: str) -> tuple[subprocess.Popen, Path]:
    """``ml-stack-serve broker`` as its own process under a state root in ``tmp_path``, once
    its record names it."""
    state = tmp_path / "broker-home"
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "ML_STACK_HOME": str(state),
           "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH")) if p)}
    proc = subprocess.Popen([sys.executable, "-m", "ml_stack.serve.cli", "broker", *extra],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    record = state / "broker.json"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and proc.poll() is None:
        try:
            if json.loads(record.read_text()).get("pid") == proc.pid:
                return proc, state
        except (OSError, ValueError):
            pass
        time.sleep(0.1)
    proc.kill()
    raise AssertionError(f"the broker did not write its record: {proc.communicate()[0]}")


@pytest.mark.slow
def test_a_broker_with_nothing_to_supervise_exits_on_its_own(tmp_path):
    proc, state = _broker_process(tmp_path, "--quit-after", "2")
    try:
        assert proc.wait(timeout=20) == 0
    finally:
        kill_process_tree(proc.pid)
    assert "nothing to supervise for 2s" in proc.stdout.read()
    assert not (state / "broker.json").exists()


@pytest.mark.slow
def test_a_broker_whose_home_is_removed_exits(tmp_path):
    import shutil

    proc, state = _broker_process(tmp_path)
    try:
        shutil.rmtree(state)
        assert proc.wait(timeout=20) == 0
    finally:
        kill_process_tree(proc.pid)
    assert "its record is gone" in proc.stdout.read()


def test_asking_for_cores_with_no_broker_starts_none():
    from ml_stack import home
    from ml_stack.testing.cores import workers_for

    assert workers_for(3) == 3
    assert not home.state("broker.json").exists()


def test_a_claim_is_supervised_and_nothing_is_not(broker, holders):
    assert not broker.supervising()
    holder = holders()
    broker.claim("gpu", holder.pid, {})
    assert broker.supervising()
    broker.unclaim("gpu", holder.pid)
    assert not broker.supervising()
