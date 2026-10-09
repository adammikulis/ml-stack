"""A server computes on one device, ``cpu`` or ``gpu``; an ask is only served by a server on
the device it names, and the device is written down, granted and listed."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pytest

from poolhouse.serve import admission, broker_cli, lease_cli, ops, status_cli, unmanaged
from poolhouse.serve.backend import LlamaServerBackend, ServerSpec
from poolhouse.serve.broker import Ask, Broker, BrokerError, Held
from poolhouse.serve.leases import recorded_servers
from poolhouse.serve.manager import ServerManager
from poolhouse.serve.process import kill_process_tree, offloaded_layers
from poolhouse.testing.fakes import FakeLlamaServer, Served, fake_llama_binary

LLAMA = {"pid": 4242, "ip": "127.0.0.1", "uid": os.getuid(), "user": "me",
         "exe": "/opt/llama.cpp/build/bin/llama-server", "rss": 1024 ** 3}


@pytest.fixture
def broker(tmp_path):
    manager = ServerManager(LlamaServerBackend(binary=fake_llama_binary(tmp_path)),
                            state_file=tmp_path / "servers.json")
    made = Broker(manager, idle_s=3600.0, room=lambda: None, scan=lambda: [])
    yield made
    for held in list(made.servers.values()):
        if held.pid and held.ours:
            kill_process_tree(held.pid)


@pytest.fixture
def model(tmp_path):
    path = tmp_path / f"only-M1-{tmp_path.name}.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 64)
    return str(path)


@pytest.fixture
def holders():
    started: list[subprocess.Popen] = []

    def one() -> subprocess.Popen:
        started.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"]))
        return started[-1]

    yield one
    for proc in started:
        proc.kill()
        proc.wait(timeout=10)


def ask(model: str, pid: int, purpose: str = "chat", **spec) -> Ask:
    return Ask(purpose=purpose, models=(model,), pid=pid, label=f"pid {pid}",
               spec={"context": 512, **spec})


def test_the_device_is_the_one_given_else_read_off_the_layers():
    assert admission.device_of(ServerSpec(model="m")) == "gpu"
    assert admission.device_of(ServerSpec(model="m", n_gpu_layers=0)) == "cpu"
    assert admission.device_of(ServerSpec(model="m", device="cpu")) == "cpu"
    assert admission.device_of({"n_gpu_layers": "0"}) == "cpu"
    assert admission.device_of({}) == "gpu"
    with pytest.raises(ValueError, match="cpu or gpu"):
        admission.device_of({"device": "tpu"})
    with pytest.raises(ValueError, match="n_gpu_layers 0"):
        admission.device_of({"device": "gpu", "n_gpu_layers": 0})


def test_a_cpu_spec_starts_a_server_with_no_layers_offloaded(tmp_path):
    backend = LlamaServerBackend(binary=fake_llama_binary(tmp_path))
    argv = backend.command(ServerSpec(model="m.gguf", device="cpu"))
    assert argv[argv.index("-ngl") + 1] == "0"
    argv = backend.command(ServerSpec(model="m.gguf"))
    assert argv[argv.index("-ngl") + 1] == "99"


def test_an_ask_names_a_device_cpu_or_gpu():
    body = {"purpose": "chat", "models": ["m.gguf"], "spec": {"device": "cpu"}}
    assert Ask.from_json(body).spec["device"] == "cpu"
    with pytest.raises(ValueError, match="cpu or gpu"):
        Ask.from_json({**body, "spec": {"device": "npu"}})


def test_a_cpu_ask_is_not_served_by_a_gpu_server_and_the_reverse():
    gpu = Held(port=1, model="m", shape={"device": "gpu", "context": 512})
    cpu = Held(port=2, model="m", shape={"device": "cpu", "context": 512})
    assert gpu.short_of({"context": 512}) == "" and cpu.short_of({"device": "cpu"}) == ""
    assert gpu.short_of({"device": "cpu"}) == "port 1 computes on gpu, cpu asked"
    assert cpu.short_of({"context": 512}) == "port 2 computes on cpu, gpu asked"
    assert cpu.short_of({"n_gpu_layers": 0}) == ""


def test_the_broker_starts_a_server_on_the_device_asked_for_and_shares_only_within_it(
        broker, model, holders):
    a, b, c = holders(), holders(), holders()
    gpu = broker.lease(ask(model, a.pid), timeout=30)
    with pytest.raises(BrokerError, match="computes on gpu, cpu asked"):
        broker.lease(ask(model, b.pid, device="cpu"), timeout=1)
    cpu = broker.lease(ask(model, b.pid, "other", device="cpu"), timeout=30)
    assert (gpu.device, cpu.device) == ("gpu", "cpu") and cpu.port != gpu.port
    again = broker.lease(ask(model, c.pid, "other", n_gpu_layers=0), timeout=30)
    assert again.port == cpu.port and again.shared and again.device == "cpu"


def test_the_device_is_in_the_lease_record(broker, model, holders):
    cpu = broker.lease(ask(model, holders().pid, device="cpu"), timeout=30)
    gpu = broker.lease(ask(model, holders().pid, "other", n_gpu_layers="auto"), timeout=30)
    records = recorded_servers(broker.manager.state_file)
    assert records[cpu.port]["device"] == "cpu" and records[gpu.port]["device"] == "gpu"


def test_a_restarted_broker_reads_each_servers_device_off_its_record(broker, model, holders):
    cpu = broker.lease(ask(model, holders().pid, device="cpu"), timeout=30)
    fresh = Broker(broker.manager, idle_s=3600.0, room=lambda: None, scan=lambda: [])
    fresh.adopt()
    assert fresh.servers[cpu.port].shape["device"] == "cpu"
    assert fresh.servers[cpu.port].short_of({"device": "gpu"})


def test_the_queue_and_the_leases_listing_name_each_servers_device(
        broker, model, holders, monkeypatch, capsys):
    cpu = broker.lease(ask(model, holders().pid, device="cpu"), timeout=30)
    monkeypatch.setattr(broker_cli.broker_wire, "status", broker.snapshot)
    monkeypatch.setattr(lease_cli, "snapshot", broker.snapshot)
    assert broker_cli.cmd_queue(type("A", (), {"json": False})()) == 0
    assert lease_cli.cmd_leases(type("A", (), {"json": False})()) == 0
    out = capsys.readouterr()
    text = out.out + out.err
    assert f":{cpu.port}" in text and "[cpu, " in text and "  cpu  1 holder(s)" in text


def test_the_command_line_says_how_many_layers_are_offloaded():
    assert offloaded_layers(["llama-server", "-ngl", "0"]) == "0"
    assert offloaded_layers(["llama-server", "--n-gpu-layers=12"]) == "12"
    assert offloaded_layers(["llama-server", "-m", "x.gguf"]) == "auto"


@pytest.mark.parametrize(("layers", "device"), [("0", "cpu"), ("99", "gpu"), ("auto", "gpu")])
def test_an_unmanaged_server_has_the_device_its_command_line_gives(layers, device):
    fake = FakeLlamaServer(Served(model="/models/m.gguf", context=4096))
    try:
        seen = unmanaged.examine(fake.port, find=lambda _port: {**LLAMA, "n_gpu_layers": layers})
        assert seen.ok and seen.device == device
        assert unmanaged.adopt_entry(fake.port, seen)["device"] == device
    finally:
        fake.close()


def test_status_names_the_device_a_recorded_server_computes_on(monkeypatch, capsys):
    fake = FakeLlamaServer(Served(model="/models/m.gguf", context=4096))
    try:
        snapshot = ops.look(fake.port, {fake.port: {"device": "cpu"}})
        assert snapshot is not None and snapshot.device == "cpu"
        found = ops.Status((fake.port,), (snapshot,), ())
        monkeypatch.setattr(ops, "status", lambda **_: found)
        monkeypatch.setattr(status_cli.holding, "holds", lambda: [])
        args = argparse.Namespace(json=None, port=fake.port, model="", context=0, parallel=1)
        status_cli.cmd_status(args)
        assert "device   cpu" in capsys.readouterr().out
    finally:
        fake.close()
