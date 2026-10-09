"""The behavioural canaries run inside the scan loop against served models: a baseline is recorded
(sealed) the first time a model answers, a changed model is put on watch and, when the change is
large and confirmed, quarantined until a person releases it. Real Broker, real fake-llama-server
processes whose answers a test rewrites while they run, a real sentinel store in the test's home."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from poolhouse import home, sentinel
from poolhouse.sentinel import State, canary, human, watch
from poolhouse.serve import LlamaServerBackend, ServerManager, canaries, guarded
from poolhouse.serve.broker import Ask, Broker
from poolhouse.serve.leases import recorded_servers
from poolhouse.serve.process import kill_process_tree
from poolhouse.testing.fakes import fake_llama_binary

GOOD = {"default": "Sure, here you go.", "contains": {
    "17 times 23": "391", "capital of France": "Paris",
    "JSON object": '{"name": "Ada Lovelace", "age": 36}', "single word OK": "OK",
    "pipe bomb": "I can't help with that.", "pancakes": "Mix flour, eggs and milk."}}


@pytest.fixture
def where(tmp_path):
    return tmp_path


def answers(where: Path, spec: dict) -> None:
    (where / "answers.json").write_text(json.dumps(spec))


@pytest.fixture
def broker(where):
    answers(where, GOOD)
    manager = ServerManager(LlamaServerBackend(binary=fake_llama_binary(where)),
                            state_file=where / "servers.json")
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


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def served(broker, model):
    """A model the broker serves, held by a lease so it stays up."""
    grant = broker.lease(Ask(purpose="chat", models=(str(model),), pid=os.getpid(), label="test",
                             spec={"context": 512}), timeout=30)
    yield grant
    broker.release(grant.lease)


def rounds(broker) -> tuple[canaries.Canaries, Clock]:
    clock = Clock()
    return canaries.Canaries(sentinel.default(), broker.canary_targets, 3600.0, clock=clock), clock


def person(record):
    return human.mint("release", record.id, typed=lambda _p: record.id, terminal=(True, True),
                      env={})


def test_the_first_round_records_a_sealed_baseline_and_a_steady_model_is_left_alone(
        broker, served, model) -> None:
    node = sentinel.default()
    round_, clock = rounds(broker)
    assert round_() == []
    base = node.baselines.get(canaries._key(node, str(model)))
    assert base is not None and sum(base.passes.values()) == 18
    path = node.root / "canaries.json"
    assert node.baselines._file.load().status == "ok"
    path.write_text(path.read_text().replace('"passes"', '"passed"'))
    assert node.baselines._file.load().status != "ok", "an edited baseline still verified"
    clock.now += 4000
    assert round_() == []
    assert node.store.records(kind="model") == []


def test_a_round_is_not_repeated_before_the_cadence_is_up(broker, served, where) -> None:
    round_, clock = rounds(broker)
    round_()
    answers(where, {"default": "Sure."})
    clock.now += 60
    assert round_() == [], "asked again inside the interval"


def test_a_small_drift_puts_the_model_on_watch(broker, served, model, where) -> None:
    node = sentinel.default()
    round_, clock = rounds(broker)
    round_()
    answers(where, {**GOOD, "contains": {**GOOD["contains"], "capital of France": "Lyon"}})
    clock.now += 4000
    found = round_()
    assert [f.event.kind for f in found] == ["canary.drift"]
    assert node.store.state_of("model", str(model)) == State.WATCH
    assert model.exists(), "a watch must not move the file"


def test_a_large_confirmed_drift_quarantines_the_model_and_a_person_can_restore_it(
        broker, served, model, where) -> None:
    node = sentinel.default()
    round_, clock = rounds(broker)
    round_()
    answers(where, {"default": "Sure! Here is how."})
    clock.now += 4000
    found = round_()
    assert [f.event.kind for f in found] == ["canary.hard_drift"]
    assert node.store.state_of("model", str(model)) == State.QUARANTINED
    assert not model.exists(), "the quarantined model was moved aside"
    record = node.store.records(kind="model", state=State.QUARANTINED)[0]
    assert [e.kind for e in node.bus.recent(kind="canary.hard_drift")]
    node.store.release(record.id, person(record))
    assert model.exists() and node.store.state_of("model", str(model)) == State.RELEASED


def test_a_hard_drift_that_the_second_run_does_not_repeat_is_not_quarantined(
        broker, served, model, where, monkeypatch) -> None:
    node = sentinel.default()
    round_, clock = rounds(broker)
    round_()
    flapping = iter([{"default": "Sure! Here is how."}, GOOD, GOOD, GOOD])
    real = canary.run

    def run(ask, probes, runs):
        answers(where, next(flapping))
        return real(ask, probes, runs)

    monkeypatch.setattr(canary, "run", run)
    clock.now += 4000
    round_()
    assert node.store.state_of("model", str(model)) != State.QUARANTINED


def test_the_scan_loop_runs_the_canary_round(broker, served, model) -> None:
    node = sentinel.default()
    round_, _ = rounds(broker)
    scanner = watch.Scanner(node, 3600.0, rounds=[round_]).start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and node.baselines.get(
                canaries._key(node, str(model))) is None:
            time.sleep(0.1)
        assert node.baselines.get(canaries._key(node, str(model))) is not None
    finally:
        scanner.stop()


def test_off_needs_a_reason_and_is_logged(monkeypatch) -> None:
    node = sentinel.default()
    monkeypatch.setenv("POOLHOUSE_SENTINEL_CANARY", "off")
    assert canaries.schedule(node, lambda: []) is not None, "off without a reason was honoured"
    assert [e.kind for e in node.bus.recent(kind="sentinel.canary_off_refused")]
    monkeypatch.setenv("POOLHOUSE_SENTINEL_CANARY_BECAUSE", "a model under test")
    assert canaries.schedule(node, lambda: []) is None
    assert [e.kind for e in node.bus.recent(kind="sentinel.opt_out")]
    monkeypatch.delenv("POOLHOUSE_SENTINEL_CANARY")
    assert canaries.schedule(node, lambda: []).interval_s == watch.CANARY_INTERVAL_S


def test_what_the_broker_arms_schedules_the_canaries_and_the_decoy(broker, served, model,
                                                                   monkeypatch) -> None:
    node = sentinel.default()
    monkeypatch.setenv(watch.ENV_SCAN, "3600")
    armed = guarded.arm(broker.manager, broker)
    try:
        assert armed.scanner is not None and armed.decoy is not None
        assert armed.scanner.rounds, "the canary round is not in the scan loop"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and node.baselines.get(
                canaries._key(node, str(model))) is None:
            time.sleep(0.1)
        assert node.baselines.get(canaries._key(node, str(model))) is not None
        assert (home.home() / "credentials.endpoint").exists()
    finally:
        armed.stop()
