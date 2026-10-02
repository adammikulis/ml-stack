"""scripts/testslots.py: a shared worker budget that makes test runs queue instead of piling up."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "testslots.py"

CHILD = r'''
import importlib.util, json, sys, time
spec = importlib.util.spec_from_file_location("testslots", sys.argv[1]); m = importlib.util.module_from_spec(spec); sys.modules["testslots"] = m; spec.loader.exec_module(m)
log, want, hold, label = sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
with m.lease(want, min(2, want), label=label, say=lambda s: None) as g:
    t0 = time.time()
    time.sleep(hold)
    with open(log, "a") as f:
        f.write(json.dumps({"label": label, "workers": g.workers, "start": t0, "end": time.time()}) + "\n")
'''


def _env(tmp_path, budget):
    return {**os.environ, "DEV_TEST_SLOTS_DIR": str(tmp_path / "slots"), "DEV_TEST_BUDGET": str(budget), "DEV_TEST_WAIT_S": "60"}


def _spawn(tmp_path, budget, want, hold, label):
    return subprocess.Popen([sys.executable, "-c", CHILD, str(SCRIPT), str(tmp_path / "log.jsonl"), str(want), str(hold), label],
                            env=_env(tmp_path, budget))


def _records(tmp_path):
    p = tmp_path / "log.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_concurrent_grants_never_exceed_the_budget_and_all_runs_finish(tmp_path):
    budget = 8
    procs = []
    for i in range(6):
        procs.append(_spawn(tmp_path, budget, 4, 0.8, f"run{i}"))
        time.sleep(0.05)
    assert all(p.wait(timeout=60) == 0 for p in procs)
    recs = _records(tmp_path)
    assert len(recs) == 6
    for r in recs:                                           # at every run's start, the workers of all live runs fit
        live = sum(o["workers"] for o in recs if o["start"] <= r["start"] < o["end"])
        assert live <= budget, (live, recs)
    assert max(r["end"] for r in recs) - min(r["start"] for r in recs) > 1.4     # they really queued (6x4 workers > 8)


def test_runs_are_granted_in_arrival_order(tmp_path):
    first = _spawn(tmp_path, 4, 4, 0.8, "first")
    time.sleep(0.3)
    second = _spawn(tmp_path, 4, 4, 0.3, "second")
    time.sleep(0.2)
    third = _spawn(tmp_path, 4, 4, 0.3, "third")
    for p in (first, second, third):
        assert p.wait(timeout=60) == 0
    order = [r["label"] for r in sorted(_records(tmp_path), key=lambda r: r["start"])]
    assert order == ["first", "second", "third"]


def test_a_killed_holder_frees_its_workers(tmp_path):
    holder = _spawn(tmp_path, 4, 4, 60, "doomed")
    deadline = time.time() + 20
    while time.time() < deadline and not list((tmp_path / "slots").glob("*.slot")):
        time.sleep(0.1)
    time.sleep(0.5)
    waiter = _spawn(tmp_path, 4, 4, 0.1, "waiter")
    time.sleep(1.0)
    assert waiter.poll() is None                              # still waiting behind the live holder
    holder.send_signal(signal.SIGKILL)
    assert waiter.wait(timeout=30) == 0
    assert [r["label"] for r in _records(tmp_path)] == ["waiter"]


def test_status_lists_running_and_waiting(tmp_path):
    holder = _spawn(tmp_path, 4, 4, 3, "holder")
    time.sleep(1.0)
    waiter = _spawn(tmp_path, 4, 4, 0.1, "waiter")
    time.sleep(1.0)
    out = subprocess.run([sys.executable, str(SCRIPT), "status"], env=_env(tmp_path, 4), capture_output=True, text=True, timeout=30).stdout
    assert "running" in out and "holder" in out and "waiting" in out and "waiter" in out
    for p in (holder, waiter):
        assert p.wait(timeout=60) == 0


def test_a_run_gets_fewer_workers_when_the_budget_is_partly_used_but_never_below_its_minimum(tmp_path):
    a = _spawn(tmp_path, 6, 4, 1.5, "a")
    time.sleep(0.6)
    b = _spawn(tmp_path, 6, 4, 0.2, "b")                       # 2 free: runs now with 2 (its minimum is min(2, 4))
    assert b.wait(timeout=30) == 0 and a.wait(timeout=30) == 0
    got = {r["label"]: r["workers"] for r in _records(tmp_path)}
    assert got == {"a": 4, "b": 2}


def test_the_queue_can_be_switched_off_explicitly(tmp_path):
    spec = importlib.util.spec_from_file_location("testslots", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["testslots"] = m
    spec.loader.exec_module(m)
    msgs: list[str] = []
    old = os.environ.get("DEV_TEST_SLOTS")
    os.environ["DEV_TEST_SLOTS"] = "off"
    try:
        with m.lease(7, label="x", say=msgs.append) as g:
            assert g.workers == 7
    finally:
        if old is None:
            os.environ.pop("DEV_TEST_SLOTS")
        else:
            os.environ["DEV_TEST_SLOTS"] = old
    assert msgs and "disabled" in msgs[0]


def test_the_run_command_queues_and_hands_the_granted_workers_to_the_command(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPT), "run", "--want", "3", "--min", "2", "--label", "t", "--",
                          sys.executable, "-c", "import os; print(os.environ['DEV_TEST_WORKERS'])"],
                         env=_env(tmp_path, 8), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.strip() == "3"
    bad = subprocess.run([sys.executable, str(SCRIPT), "run", "--want", "1", "--", sys.executable, "-c", "raise SystemExit(7)"],
                         env=_env(tmp_path, 8), capture_output=True, text=True, timeout=60)
    assert bad.returncode == 7                                   # the command's exit status is passed through


def _load():
    spec = importlib.util.spec_from_file_location("testslots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["testslots"] = module
    spec.loader.exec_module(module)
    return module


def test_the_cap_follows_the_load_average():
    m = _load()
    # 16 cores: base 12; an idle machine may use all 16; a saturated one backs off to half the base
    assert m.cap_for(None, cores=16) == 12
    assert m.cap_for(2.0, cores=16) == 16
    assert m.cap_for(11.9, cores=16) == 16                    # just under 0.75 * cores
    assert m.cap_for(12.0, cores=16) == 12
    assert m.cap_for(31.9, cores=16) == 12                    # just under 2 * cores
    assert m.cap_for(32.1, cores=16) == 6
    assert m.cap_for(500.0, cores=4) == 2 and m.cap_for(0.1, cores=4) == 4   # a floor on small machines


def test_a_pinned_budget_ignores_the_load(monkeypatch):
    m = _load()
    monkeypatch.setenv("DEV_TEST_BUDGET", "5")
    assert m.budget() == 5


HEAVY_CHILD = r'''
import importlib.util, json, sys, time
spec = importlib.util.spec_from_file_location("testslots", sys.argv[1]); m = importlib.util.module_from_spec(spec); sys.modules["testslots"] = m; spec.loader.exec_module(m)
log = sys.argv[2]
with m.heavy_lane("t", say=lambda s: None):
    t0 = time.time(); time.sleep(float(sys.argv[3]))
    with open(log, "a") as f:
        f.write(json.dumps({"start": t0, "end": time.time()}) + "\n")
'''


def test_heavy_lanes_limit_concurrent_heavy_work_across_processes(tmp_path):
    env = {**_env(tmp_path, 8), "DEV_TEST_HEAVY_LANES": "2"}
    procs = [subprocess.Popen([sys.executable, "-c", HEAVY_CHILD, str(SCRIPT), str(tmp_path / "lanes.jsonl"), "0.6"],
                              env=env) for _ in range(6)]
    assert all(p.wait(timeout=60) == 0 for p in procs)
    recs = [json.loads(x) for x in (tmp_path / "lanes.jsonl").read_text().splitlines()]
    assert len(recs) == 6
    for r in recs:
        assert sum(1 for o in recs if o["start"] <= r["start"] + 1e-3 < o["end"]) <= 2, recs


def test_a_killed_heavy_test_frees_its_lane(tmp_path):
    env = {**_env(tmp_path, 8), "DEV_TEST_HEAVY_LANES": "1"}
    holder = subprocess.Popen([sys.executable, "-c", HEAVY_CHILD, str(SCRIPT), str(tmp_path / "lanes.jsonl"), "60"],
                              env=env)
    time.sleep(1.0)
    waiter = subprocess.Popen([sys.executable, "-c", HEAVY_CHILD, str(SCRIPT), str(tmp_path / "lanes.jsonl"), "0.1"],
                              env=env)
    time.sleep(1.0)
    assert waiter.poll() is None
    holder.send_signal(signal.SIGKILL)
    assert waiter.wait(timeout=30) == 0
    holder.wait(timeout=10)
