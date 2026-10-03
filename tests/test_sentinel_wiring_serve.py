"""The Broker starts no server for a model sentinel holds or that no longer matches its pin, a
daemon runs the periodic scan, and `ml-stack-security status` says whether it is armed. Real
Broker, real fake-llama-server processes, real sentinel store under the test's state root."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from ml_stack import home, sentinel
from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.daemon import load_or_create_token
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.remote import Peer, PeerError
from ml_stack.http import Server
from ml_stack.sentinel import State, human, watch
from ml_stack.sentinel.cli import status as security_status
from ml_stack.sentinel.store import Holding
from ml_stack.serve import LlamaServerBackend, ServerManager, ServerSpec
from ml_stack.serve.broker import Ask, Broker, BrokerError
from ml_stack.serve.guarded import SentinelRefused
from ml_stack.serve.leases import recorded_servers
from ml_stack.serve.ports import free_port
from ml_stack.serve.process import kill_process_tree, pid_exists
from ml_stack.testing.fakes import fake_llama_binary

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def manager(tmp_path):
    return ServerManager(LlamaServerBackend(binary=fake_llama_binary(tmp_path)),
                         state_file=tmp_path / "servers.json")


@pytest.fixture
def broker(manager):
    made = Broker(manager, idle_s=3600.0, room=lambda: None, scan=lambda: [])
    yield made
    for held in list(made.servers.values()):
        if held.pid and held.ours:
            kill_process_tree(held.pid)
    for entry in recorded_servers(manager.state_file).values():
        if entry.get("pid"):
            kill_process_tree(entry["pid"])


@pytest.fixture
def model(tmp_path):
    folder = home.home() / "models"
    folder.mkdir(parents=True)
    path = folder / f"m-{tmp_path.name}.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 4096)
    return path


def ask(path: Path) -> Ask:
    return Ask(purpose="chat", models=(str(path),), pid=os.getpid(), label="test",
               spec={"context": 512})


def person(record):
    return human.mint("release", record.id, typed=lambda _p: record.id, terminal=(True, True),
                      env={})


def test_an_unpinned_model_is_pinned_on_first_use_and_served(broker, model):
    grant = broker.lease(ask(model), timeout=30)
    pin = sentinel.default().manifest.pins()[str(model)]
    assert pin.source == "first-use" and pin.kind == "model"
    assert [e.kind for e in sentinel.default().bus.recent(kind="model.pinned")]
    assert grant.port in broker.servers
    broker.release(grant.lease)


def test_a_tampered_pinned_model_is_refused_and_no_server_starts(broker, manager, model):
    node = sentinel.default()
    node.manifest.pin(model, "model", source="hf://org/repo@rev")
    with model.open("r+b") as fh:
        fh.seek(2000)
        fh.write(b"\xff")
    with pytest.raises(BrokerError, match="pin"):
        broker.lease(ask(model), timeout=30)
    assert broker.servers == {} and recorded_servers(manager.state_file) == {}
    assert node.store.state_of("model", str(model)) == State.QUARANTINED
    assert not model.exists(), "the tampered file was left where it can be loaded"
    with pytest.raises(BrokerError, match="quarantined"):
        broker.lease(ask(model), timeout=30)


def test_a_tampered_model_is_refused_through_the_manager_too(manager, model):
    node = sentinel.default()
    node.manifest.pin(model, "model", source="hf://org/repo@rev")
    model.write_bytes(b"GGUF" + b"\x01" * 4096)
    with pytest.raises(SentinelRefused, match="pin"):
        manager.lease(ServerSpec(model=str(model), port=free_port()), roam=False, timeout=5)
    assert recorded_servers(manager.state_file) == {}


def test_a_quarantined_model_cannot_be_leased_until_a_person_releases_it(broker, model):
    node = sentinel.default()
    held = node.store.quarantine(("model", str(model)), "seen misbehaving", {}, Holding())
    with pytest.raises(BrokerError, match="quarantined"):
        broker.lease(ask(model), timeout=30)
    with pytest.raises(human.HumanRequired):
        human.mint("release", held.id, terminal=(False, False), env={})
    with pytest.raises(human.HumanRequired):
        human.mint("release", held.id, typed=lambda _p: held.id, terminal=(True, True),
                   env={"ML_STACK_AGENT": "1"})
    with pytest.raises(BrokerError, match="quarantined"):
        broker.lease(ask(model), timeout=30)
    node.store.release(held.id, person(held))
    grant = broker.lease(ask(model), timeout=30)
    assert grant.port in broker.servers
    broker.release(grant.lease)


def test_a_quarantined_model_is_not_shared_from_a_server_that_is_still_up(broker, model):
    first = broker.lease(ask(model), timeout=30)
    node = sentinel.default()
    node.store.on_quarantine["model"] = []
    node.store.quarantine(("model", str(model)), "seen misbehaving", {}, Holding())
    assert pid_exists(broker.servers[first.port].pid)
    with pytest.raises(BrokerError, match="quarantined"):
        broker.lease(ask(model), timeout=30)
    assert len(broker.servers[first.port].holders) == 1


def test_quarantining_a_model_stops_the_servers_started_for_it(broker, manager, model):
    grant = broker.lease(ask(model), timeout=30)
    pid = broker.servers[grant.port].pid
    assert pid_exists(pid)
    sentinel.default().store.quarantine(("model", str(model)), "pin mismatch", {}, Holding())
    deadline = time.monotonic() + 15
    while pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not pid_exists(pid), "the quarantined model's server kept running"


def test_a_scan_loop_finds_a_pinned_file_changed_while_it_runs(model, monkeypatch):
    monkeypatch.setenv(watch.ENV_SCAN, "0.2")
    node = sentinel.default()
    node.manifest.pin(model, "model", source="test")
    loop = sentinel.arm_scan(node)
    assert loop is not None
    try:
        assert watch.scanner_state(node.root)["armed"]
        model.write_bytes(b"GGUF" + b"\x02" * 9000)
        deadline = time.monotonic() + 10
        while (node.store.state_of("model", str(model)) != State.QUARANTINED
               and time.monotonic() < deadline):
            time.sleep(0.1)
        assert node.store.state_of("model", str(model)) == State.QUARANTINED
    finally:
        loop.stop()
    assert not watch.scanner_state(node.root)["armed"]


def test_status_says_whether_the_loop_is_armed(model, monkeypatch, capsys):
    monkeypatch.setenv(watch.ENV_SCAN, "30")
    node = sentinel.default()
    security_status(argparse.Namespace(json=True))
    assert json.loads(capsys.readouterr().out)["scanner"]["armed"] is False
    loop = sentinel.arm_scan(node)
    try:
        security_status(argparse.Namespace(json=True))
        assert json.loads(capsys.readouterr().out)["scanner"]["armed"] is True
        security_status(argparse.Namespace(json=False))
        assert "scanner: armed" in capsys.readouterr().out
    finally:
        loop.stop()
    security_status(argparse.Namespace(json=False))
    assert "NOT ARMED" in capsys.readouterr().out


def test_the_scan_is_switched_off_only_with_a_reason_and_it_is_logged(monkeypatch):
    node = sentinel.default()
    monkeypatch.setenv(watch.ENV_SCAN, "off")
    monkeypatch.delenv(watch.BECAUSE, raising=False)
    refused = sentinel.arm_scan(node)
    assert refused is not None, "off without a reason switched the scan off"
    refused.stop()
    assert node.bus.recent(kind="sentinel.scan_off_refused")
    monkeypatch.setenv(watch.BECAUSE, "a laptop on battery for a flight")
    assert sentinel.arm_scan(node) is None
    got = node.bus.recent(kind="sentinel.opt_out")
    assert got and got[-1].evidence["because"] == "a laptop on battery for a flight"
    assert got[-1].subject == "opt_out:the periodic scan"


def test_the_broker_daemon_runs_the_scan_and_stops_it_on_exit(model):
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), watch.ENV_SCAN: "0.3"}
    daemon = subprocess.Popen([sys.executable, "-m", "ml_stack.serve.cli", "broker",
                               "--quit-after", "120"], env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    root = sentinel.default().root
    try:
        deadline = time.monotonic() + 30
        while not watch.scanner_state(root)["armed"] and time.monotonic() < deadline:
            time.sleep(0.2)
        state = watch.scanner_state(root)
        assert state["armed"] and state["pid"] == daemon.pid, state
    finally:
        daemon.terminate()
        daemon.wait(timeout=30)
    assert not watch.scanner_state(root)["armed"]


def test_the_fleet_daemon_quarantines_a_peer_that_forges_requests(tmp_path):
    root = tmp_path / "traind"
    (root / "files").mkdir(parents=True)
    token = load_or_create_token(root)
    runner = JobRunner(root)
    httpd = Server(("127.0.0.1", 0), make_handler(Daemon(runner, root / "files", token)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        forger = Peer(url, "not-the-token")
        for _ in range(60):
            with pytest.raises(PeerError):
                forger.jobs()
        assert sentinel.default().store.state_of("peer", "127.0.0.1") == State.QUARANTINED
        with pytest.raises(PeerError, match=r"quarantined|locked|429|401"):
            Peer(url, token).jobs()
    finally:
        runner.shutdown()
        httpd.shutdown()
        httpd.server_close()
