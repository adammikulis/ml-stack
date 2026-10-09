"""Each flagged feature is absent while it is off and present once it is on."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from poolhouse import features, node_binary, node_health
from poolhouse.fleet.settings import Settings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import testextras


@pytest.fixture
def machine(monkeypatch, tmp_path):
    monkeypatch.setenv("POOLHOUSE_HOME", str(tmp_path / "state"))


@pytest.fixture
def on_windows(monkeypatch):
    for module in (node_binary, node_health):
        monkeypatch.setattr(module, "is_windows", lambda: True)


STATE = Path("/tmp/node-state")


def test_on_windows_the_node_takes_the_exe_and_the_pipe_with_no_flag(machine, on_windows):
    assert node_binary.name() == "poolhouse-node.exe"
    assert str(node_health.socket_path(STATE)).startswith("\\\\.\\pipe\\poolhouse-node-")


def test_elsewhere_the_node_takes_the_plain_binary_and_the_socket(machine, monkeypatch):
    for module in (node_binary, node_health):
        monkeypatch.setattr(module, "is_windows", lambda: False)
    assert node_binary.name() == "poolhouse-node"
    assert node_health.socket_path(STATE) == STATE / node_health.SOCKET


def test_the_test_runner_extras_are_absent_until_enabled(machine, tmp_path, monkeypatch):
    history = {"tests/test_a.py::t": {"cpu": 1.0, "wall": 2.0, "n": 3}}
    monkeypatch.setattr(testextras.testhistory, "load", lambda path: history)
    environment, plugins = {}, []
    assert testextras.on() is False
    assert testextras.learned_order(tmp_path, environment, plugins) is None
    assert environment == {} and plugins == []


def test_the_test_runner_extras_order_by_the_recorded_times_once_enabled(machine, tmp_path, monkeypatch):
    history = {"tests/test_a.py::t": {"cpu": 1.0, "wall": 2.0, "n": 3}}
    monkeypatch.setattr(testextras.testhistory, "load", lambda path: history)
    monkeypatch.setattr(testextras.testhistory, "history_path", lambda root: root / "history.json")
    features.switch("test-runner-extras", True)
    environment, plugins = {}, []
    order = testextras.learned_order(tmp_path, environment, plugins)
    try:
        assert order and environment["DEV_TEST_ORDER"] == order and plugins == ["-p", "testorder"]
    finally:
        Path(order).unlink(missing_ok=True)


def test_the_runner_only_records_the_full_tier_time_through_the_extras():
    text = (Path(__file__).resolve().parent.parent / "scripts" / "test").read_text()
    assert "if testextras.on() and tier in" in text and "testextras.learned_order(" in text
    assert "import testorder" not in text


def test_guard_change_is_registered_with_no_gate_to_flip(machine):
    assert features.FEATURES["guard-change"].stage == "experimental"
    assert features.enabled("guard-change") is False
    features.switch("guard-change", True)
    assert features.enabled("guard-change") is True
    gated = [p for p in Path(features.__file__).parent.rglob("*.py")
             if "enabled(\"guard-change\")" in p.read_text() or "enabled('guard-change')" in p.read_text()]
    assert gated == []


def test_a_settings_save_keeps_the_files_own_features_never_the_objects(tmp_path):
    path = tmp_path / "settings.json"
    Settings(features={"test-runner-extras": True}).save(path)
    assert Settings.load(path).features == {}   # a save keeps the file's own map, never the object's
