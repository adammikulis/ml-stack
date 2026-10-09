"""The workspace load harness: a small real run, and the budgets it judges against."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from workspace_kit import Kit, clean_env

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/experiments/workspace_load.py"
spec = importlib.util.spec_from_file_location("workspace_load", SCRIPT)
load = importlib.util.module_from_spec(spec)
sys.modules["workspace_load"] = load
spec.loader.exec_module(load)


def run_script(*args: str) -> tuple[int, dict]:
    env = {k: v for k, v in os.environ.items() if k not in ("ML_STACK_WORKSPACE_HOME", "ML_STACK_HOME")}
    done = subprocess.run([sys.executable, str(SCRIPT), *args], env=env, capture_output=True, text=True,
                          timeout=300, check=False)
    assert done.stdout.strip(), done.stderr
    return done.returncode, json.loads(done.stdout)


def test_a_small_run_delivers_everything_in_one_verifiable_chain_with_one_master_create():
    code, out = run_script("--agents", "6", "--messages", "4", "--drain", "30")
    assert code == 0 and out["pass"] is True and out["missed"] == [], (out["missed"], out["budgets"], out["send_s"], out["delivery_s"], out["cpu_s_per_agent_per_message"])
    assert out["sent"] == out["received"] == 24 and out["failures"] == {}
    assert out["log"]["chain_ok"] is True and out["log"]["bus_rows"] == 24
    assert out["keystore_backend_calls"]["set"] == 1 and out["keystore_backend_calls"]["get"] <= 7
    assert out["bus_lock_hold_s"]["n"] >= 24 and out["delivery_s"]["n"] == 24
    assert out["machine"]["cpus"] and out["date"]


def test_the_shipped_rate_limit_shows_up_as_counted_failures_not_a_failed_run():
    code, out = run_script("--agents", "3", "--messages", "34", "--drain", "10", "--default-limits")
    # The limit is 28 sends in a 60 second window: a host so slow that the run outlasts half a
    # window sends under it, so only the count of sends and refusals together is fixed there.
    limited = out["failures"].get("rate-limited", 0)
    assert out["sent"] + limited == 3 * 34 and code == 0 and out["pass"] is True, (
        out["missed"], out["delivery_s"], out["cpu_s_per_agent_per_message"])
    if out["wall_s"] < 30:
        assert out["failures"] == {"rate-limited": 3 * 6} and out["sent"] == 3 * 28


@pytest.fixture
def kit(monkeypatch, tmp_path):
    return Kit(clean_env(monkeypatch, tmp_path))


def synthetic(**more) -> dict:
    base = {"send": [0.01], "inbox": [0.01], "delivery": [0.01], "heartbeat": [], "thread": [], "claim": [],
            "keystore": [0.01], "lock_wait": [0.0], "lock_hold": [0.0], "failures": {}, "received": 1,
            "sent": 1, "cpu_s": 0.01}
    return {**base, **more}


@pytest.mark.parametrize(("change", "missed"), [
    ({}, []),
    ({"send": [5.0]}, ["send_p99_s"]),
    ({"delivery": [9.0]}, ["delivery_p99_s"]),
    ({"inbox": [3.0]}, ["inbox_read_p99_s"]),
    ({"lock_wait": [4.0]}, ["lock_wait_p99_s"]),
    ({"failures": {"busy": 1}}, ["keystore_busy", "unexpected_failures"]),
    ({"failures": {"denied": 2}}, ["unexpected_failures"]),
    ({"received": 0}, ["lost_messages"]),
    ({"cpu_s": 99.0}, ["cpu_s_per_agent_per_message"]),
])
def test_each_budget_fails_the_run_when_its_measure_is_over(kit, tmp_path, change, missed):
    kit.agent("peer")
    kit.ws.send(kit.owner, "peer", "note", "x")
    ring = tmp_path / "ring.json"
    shape = {"agents": 1, "messages": 1, "wall": 1.0, "default_limits": False}
    out = load.report([synthetic(**change)], kit.ws, ring, shape)
    assert out["missed"] == missed and out["pass"] is (not missed)


def test_a_broken_chain_fails_the_run(kit, tmp_path):
    kit.agent("peer")
    kit.ws.send(kit.owner, "peer", "note", "first")
    kit.ws.send(kit.owner, "peer", "note", "pay the invoice")
    with kit.ws.bus.log.graph.opened() as graph:
        event = kit.ws.bus.log.graph._events(graph, "bus")[-1]
        event["row"]["body"] = "pay the other invoice"
        graph.upsert_node(event)
    shape = {"agents": 1, "messages": 1, "wall": 1.0, "default_limits": False}
    out = load.report([synthetic()], kit.ws, tmp_path / "ring.json", shape)
    assert "chain" in out["missed"] and out["pass"] is False


def test_the_percentile_helpers():
    assert load.pct([], 0.5) == 0.0 and load.pct([3, 1, 2], 0.5) == 2 and load.pct(list(range(100)), 0.99) == 99
    assert load.summary([1.0, 2.0])["max"] == 2.0
