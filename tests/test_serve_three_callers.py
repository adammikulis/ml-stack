"""Three callers, three models, one machine: the broker holds all three leases.

Each caller is its own process calling `ServerManager.lease` with nothing else set. The
broker (a real daemon process) starts the servers, records who holds each, rates their
memory together, and queues the requests sent to them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ml_stack import gate, limits
from ml_stack.http import request_json
from ml_stack.serve import admission, broker_wire
from ml_stack.serve.leases import lease_file, recorded_servers
from ml_stack.serve.process import kill_process_tree, pid_exists
from ml_stack.testing.fakes import fake_llama_binary

pytestmark = pytest.mark.slow
SRC = str(Path(__file__).resolve().parent.parent / "src")
GIB = 1024 ** 3

CALLER = """
import json, sys, time
import ml_stack.serve.unmanaged as unmanaged
unmanaged.every_server = lambda: []
from ml_stack.serve import ServerManager, ServerSpec, free_port
from ml_stack.serve.backend import ServerFailed
manager = ServerManager()
try:
    info = manager.lease(ServerSpec(model=sys.argv[1], port=free_port(), context=512),
                         preflight=False, timeout=60)
except ServerFailed as why:
    print(json.dumps({"refused": type(why).__name__, "said": str(why)}), flush=True)
    raise SystemExit(0)
print(json.dumps({"port": info.port, "pid": info.pid, "adopted": info.adopted,
                  "lease": info.lease}), flush=True)
time.sleep(float(sys.argv[2]))
"""


DAEMON = """
import ml_stack.serve.unmanaged as unmanaged
from ml_stack.serve import broker, broker_wire
unmanaged.every_server = lambda: []
made = broker.Broker.__init__


def init(self, *args, **kwargs):
    made(self, *args, **kwargs)
    self.scan = lambda: []


broker.Broker.__init__ = init
broker_wire.serve()
"""


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """This machine's broker, running as it would, over a machine with no other servers."""
    binary = fake_llama_binary(tmp_path)
    monkeypatch.delenv(broker_wire.LOCAL_ENV)
    monkeypatch.setenv("LLAMA_CPP_SERVER", str(binary))
    monkeypatch.setenv("PYTHONPATH", SRC)
    monkeypatch.setenv(admission.ENV_WAIT, "2")
    limits.changed(memory_bytes=10 * GIB)
    daemon = subprocess.Popen([sys.executable, "-c", DAEMON], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and broker_wire.running() != daemon.pid:
        time.sleep(0.2)
    assert broker_wire.running() == daemon.pid
    yield
    try:
        for server in broker_wire.status()["servers"]:
            if server["pid"]:
                kill_process_tree(server["pid"])
    except (OSError, ValueError, broker_wire.BrokerError):
        pass
    kill_process_tree(daemon.pid)


def model(tmp_path: Path, name: str, gib: float) -> str:
    path = tmp_path / name
    with path.open("wb") as handle:
        handle.truncate(int(gib * GIB))
    return str(path)


def caller(path: str, hold: float = 60) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", CALLER, path, str(hold)],
                            stdout=subprocess.PIPE, text=True)


def first_line(proc: subprocess.Popen) -> dict:
    assert proc.stdout is not None
    return json.loads(proc.stdout.readline())


def test_three_callers_three_models_one_broker_holding_every_lease(tmp_path, machine):
    models = [model(tmp_path, f"m{n}.gguf", 2) for n in range(3)]
    procs = [caller(path) for path in models]
    try:
        got = [first_line(proc) for proc in procs]
        assert all("port" in one for one in got), got
        assert len({one["port"] for one in got}) == 3, "three servers, one per model"
        daemon = broker_wire.running()
        assert daemon and pid_exists(daemon)

        records = recorded_servers(lease_file())
        assert sorted(records) == sorted(one["port"] for one in got)
        assert {entry["owner_pid"] for entry in records.values()} == {daemon}, (
            "every server is the broker's, whoever asked for it")
        assert {entry["device"] for entry in records.values()} == {"gpu"}

        seen = {s["port"]: s for s in broker_wire.status()["servers"]}
        for proc, one in zip(procs, got, strict=True):
            assert [h["pid"] for h in seen[one["port"]]["holders"]] == [proc.pid]
        held = json.loads(lease_file().with_name("broker-leases.json").read_text())
        assert len(held) == 3, "broker-leases.json lists all three"

        for one in got:
            assert gate.device_of(f"http://127.0.0.1:{one['port']}/v1/chat/completions") == "gpu"
            assert request_json(f"http://127.0.0.1:{one['port']}/v1/chat/completions",
                                payload={"messages": [{"role": "user", "content": "hi"}]},
                                timeout=30)
    finally:
        for proc in procs:
            proc.kill()
            proc.wait(timeout=10)


def test_a_caller_that_would_not_fit_is_refused_with_the_holders_named(tmp_path, machine):
    first, second = model(tmp_path, "a.gguf", 6), model(tmp_path, "b.gguf", 6)
    holder = caller(first)
    try:
        assert "port" in first_line(holder)
        late = caller(second)
        refusal = first_line(late)
        assert refusal["refused"] == "AdmissionRefused"
        assert "red" in refusal["said"] and "a.gguf" in refusal["said"]
        late.wait(timeout=30)
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_a_second_caller_for_the_same_model_shares_the_server_and_a_dead_one_lets_go(
        tmp_path, machine):
    path = model(tmp_path, "m.gguf", 2)
    first, second = caller(path), caller(path)
    try:
        a, b = first_line(first), first_line(second)
        assert a["port"] == b["port"] and (a["adopted"] or b["adopted"])
        held = next(s for s in broker_wire.status()["servers"] if s["port"] == a["port"])
        assert {h["pid"] for h in held["holders"]} == {first.pid, second.pid}

        first.kill()
        first.wait(timeout=10)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            held = next(s for s in broker_wire.status()["servers"] if s["port"] == a["port"])
            if [h["pid"] for h in held["holders"]] == [second.pid]:
                break
            time.sleep(0.3)
        assert [h["pid"] for h in held["holders"]] == [second.pid], "the broker dropped the dead holder"
        assert pid_exists(a["pid"]), "the server stays up for the one that still holds it"
    finally:
        for proc in (first, second):
            proc.kill()
            proc.wait(timeout=10)
