"""scripts/testslots.py: class and estimate-based admission, with real lease processes."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SCRIPT = SCRIPTS / "testslots.py"

CHILD = r'''
import importlib.util, json, sys, time
from pathlib import Path
spec = importlib.util.spec_from_file_location("testslots", sys.argv[1]); m = importlib.util.module_from_spec(spec); sys.modules["testslots"] = m; spec.loader.exec_module(m)
log, want, label, klass, release, hold = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], float(sys.argv[7])
with m.lease(want, 1, label=label, say=lambda s: None, run_class=klass) as g:
    t0 = time.time()
    deadline = time.monotonic() + 40
    if release:
        while not Path(release).exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
    else:
        time.sleep(hold)
    Path(log).write_text(json.dumps({"label": label, "workers": g.workers, "start": t0, "end": time.time()}))
'''


def _env(tmp_path, budget, **extra):
    inherited = {k: v for k, v in os.environ.items() if k not in {"DEV_TEST_REMOTE_LEASE", "DEV_TEST_LEASE", "DEV_TEST_CLASS", "DEV_TEST_ESTIMATE_S", "DEV_TEST_HISTORY"}}
    return {**inherited, "DEV_TEST_SLOTS_DIR": str(tmp_path / "slots"), "DEV_TEST_BUDGET": str(budget),
            "DEV_TEST_WAIT_S": "60", "DEV_TEST_NORMAL_HOURS": "off", **extra}


def _spawn(tmp_path, budget, want, label, klass, **extra):
    release, hold = extra.pop("release", None), extra.pop("hold", 0.0)
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    return subprocess.Popen([sys.executable, "-c", CHILD, str(SCRIPT), str(records / f"{label}.json"), str(want),
                             label, klass, str(release or ""), str(hold)], env=_env(tmp_path, budget, **extra))


def _module():
    spec = importlib.util.spec_from_file_location("testslots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["testslots"] = module
    spec.loader.exec_module(module)
    return module


def _slot(tmp_path, label):
    module = _module()
    (tmp_path / "slots").mkdir(exist_ok=True)
    with module._mutex(tmp_path / "slots"):
        return next((slot for slot in module._read(tmp_path / "slots") if slot.data["label"] == label), None)


def _wait_for(tmp_path, process, label, granted):
    deadline = time.monotonic() + 20
    while True:
        assert process.poll() is None, label
        slot = _slot(tmp_path, label)
        if slot is not None and slot.granted == granted:
            return slot
        assert time.monotonic() < deadline, (label, granted)
        time.sleep(.01)


def _record(tmp_path, label):
    return json.loads((tmp_path / "records" / f"{label}.json").read_text())


def test_an_interactive_request_is_granted_before_an_older_queued_background_request(tmp_path):
    gate = tmp_path / "gate"
    holder = _spawn(tmp_path, 2, 2, "holder", "interactive", release=gate)
    _wait_for(tmp_path, holder, "holder", 2)
    background = _spawn(tmp_path, 2, 2, "bg", "background", hold=0.3)
    _wait_for(tmp_path, background, "bg", 0)
    interactive = _spawn(tmp_path, 2, 2, "fg", "interactive", hold=0.3)
    _wait_for(tmp_path, interactive, "fg", 0)
    gate.write_text("go")
    assert all(p.wait(timeout=60) == 0 for p in (holder, background, interactive))
    assert _record(tmp_path, "fg")["start"] < _record(tmp_path, "bg")["start"]


def test_background_runs_hold_at_most_half_the_budget_while_an_interactive_run_is_active(tmp_path):
    gate, bg_gate = tmp_path / "gate", tmp_path / "bg-gate"
    interactive = _spawn(tmp_path, 6, 1, "fg", "interactive", release=gate)
    _wait_for(tmp_path, interactive, "fg", 1)
    first = _spawn(tmp_path, 6, 6, "bg1", "background", release=bg_gate)
    assert _wait_for(tmp_path, first, "bg1", 3).granted == 3            # ceil(6 / 2), not the 5 left free
    second = _spawn(tmp_path, 6, 1, "bg2", "background", hold=0.1)
    _wait_for(tmp_path, second, "bg2", 0)
    time.sleep(0.5)
    assert _slot(tmp_path, "bg2").granted == 0                        # two workers are free but the class has its half
    gate.write_text("go")                                             # the interactive run ends: the cap lifts
    assert second.wait(timeout=60) == 0
    bg_gate.write_text("go")
    assert first.wait(timeout=60) == 0 and interactive.wait(timeout=60) == 0


def test_background_alone_may_use_the_whole_budget(tmp_path):
    gate = tmp_path / "gate"
    only = _spawn(tmp_path, 6, 6, "bg", "background", release=gate)
    assert _wait_for(tmp_path, only, "bg", 6).granted == 6
    gate.write_text("go")
    assert only.wait(timeout=60) == 0


def test_a_background_request_is_granted_after_the_bounded_wait_despite_the_cap(tmp_path):
    gate, bg_gate = tmp_path / "gate", tmp_path / "bg-gate"
    extra = {"DEV_TEST_BACKGROUND_WAIT_S": "2"}
    interactive = _spawn(tmp_path, 4, 1, "fg", "interactive", release=gate, **extra)
    _wait_for(tmp_path, interactive, "fg", 1)
    first = _spawn(tmp_path, 4, 2, "bg1", "background", release=bg_gate, **extra)
    _wait_for(tmp_path, first, "bg1", 2)
    started = time.monotonic()
    starved = _spawn(tmp_path, 4, 1, "bg2", "background", hold=0.1, **extra)
    _wait_for(tmp_path, starved, "bg2", 0)
    time.sleep(1.0)
    assert _slot(tmp_path, "bg2").granted == 0                        # inside the bound the cap still holds
    assert starved.wait(timeout=60) == 0                              # past it, granted with the interactive run still live
    assert time.monotonic() - started >= 2
    assert interactive.poll() is None and first.poll() is None
    gate.write_text("go")
    bg_gate.write_text("go")
    assert interactive.wait(timeout=60) == 0 and first.wait(timeout=60) == 0


def test_status_prints_each_leases_class(tmp_path):
    gate = tmp_path / "gate"
    holder = _spawn(tmp_path, 2, 2, "holder-fg", "interactive", release=gate)
    _wait_for(tmp_path, holder, "holder-fg", 2)
    waiter = _spawn(tmp_path, 2, 1, "waiter-bg", "background", hold=0.1)
    _wait_for(tmp_path, waiter, "waiter-bg", 0)
    out = subprocess.run([sys.executable, str(SCRIPT), "status"], env=_env(tmp_path, 2), capture_output=True,
                         text=True, timeout=30).stdout
    gate.write_text("go")
    assert any("running" in line and "interactive" in line and "holder-fg" in line for line in out.splitlines()), out
    assert any("waiting" in line and "background" in line and "waiter-bg" in line for line in out.splitlines()), out
    assert holder.wait(timeout=60) == 0 and waiter.wait(timeout=60) == 0


NICE_CHILD = r'''
import os, sys
sys.path.insert(0, sys.argv[1])
import testslots_runner
code = "import os, sys; open(sys.argv[1], 'w').write(str(os.nice(0)))"
status = testslots_runner.run_pytest([sys.executable, "-c", code, sys.argv[3]], 1, "nice probe",
                                     {**os.environ, "DEV_TEST_CLASS": sys.argv[2]})
sys.exit(status)
'''


@pytest.mark.skipif(not hasattr(os, "nice"), reason="no os.nice on this platform")
def test_a_background_child_runs_at_lower_cpu_priority(tmp_path):
    seen = {}
    for klass in ("interactive", "background"):
        out = tmp_path / f"{klass}.nice"
        done = subprocess.run([sys.executable, "-c", NICE_CHILD, str(SCRIPTS), klass, str(out)],
                              env=_env(tmp_path, 4), capture_output=True, text=True, timeout=60, cwd=ROOT)
        assert done.returncode == 0, done.stderr
        seen[klass] = int(out.read_text())
    assert seen["background"] == min(seen["interactive"] + 10, 19) > seen["interactive"], seen


def test_a_short_request_queued_after_a_long_one_starts_first(tmp_path):
    gate = tmp_path / "gate"
    holder = _spawn(tmp_path, 1, 1, "holder", "interactive", release=gate)
    _wait_for(tmp_path, holder, "holder", 1)
    long = _spawn(tmp_path, 1, 1, "long", "interactive", hold=0.3, DEV_TEST_ESTIMATE_S="1000")
    _wait_for(tmp_path, long, "long", 0)
    short = _spawn(tmp_path, 1, 1, "short", "interactive", hold=0.3, DEV_TEST_ESTIMATE_S="10")
    _wait_for(tmp_path, short, "short", 0)
    gate.write_text("go")
    assert all(p.wait(timeout=60) == 0 for p in (holder, long, short))
    assert _record(tmp_path, "short")["start"] < _record(tmp_path, "long")["start"]


def test_aging_grants_a_long_request_ahead_of_a_newer_short_one(tmp_path):
    gate = tmp_path / "gate"
    extra = {"DEV_TEST_BACKGROUND_WAIT_S": "2"}
    holder = _spawn(tmp_path, 1, 1, "holder", "interactive", release=gate, **extra)
    _wait_for(tmp_path, holder, "holder", 1)
    long = _spawn(tmp_path, 1, 1, "long", "interactive", hold=0.3, DEV_TEST_ESTIMATE_S="1000", **extra)
    _wait_for(tmp_path, long, "long", 0)
    time.sleep(2.5)
    short = _spawn(tmp_path, 1, 1, "short", "interactive", hold=0.3, DEV_TEST_ESTIMATE_S="10", **extra)
    _wait_for(tmp_path, short, "short", 0)
    gate.write_text("go")
    assert all(p.wait(timeout=60) == 0 for p in (holder, long, short))
    assert _record(tmp_path, "long")["start"] < _record(tmp_path, "short")["start"]


def test_the_cap_is_lifted_outside_normal_hours(tmp_path):
    gate, bg_gate = tmp_path / "gate", tmp_path / "bg-gate"
    extra = {"DEV_TEST_NORMAL_HOURS": "00:00-00:01"}
    interactive = _spawn(tmp_path, 6, 1, "fg", "interactive", release=gate, **extra)
    _wait_for(tmp_path, interactive, "fg", 1)
    wide = _spawn(tmp_path, 6, 6, "bg", "background", release=bg_gate, **extra)
    assert _wait_for(tmp_path, wide, "bg", 5).granted == 5            # everything free, not half the budget
    gate.write_text("go")
    bg_gate.write_text("go")
    assert interactive.wait(timeout=60) == 0 and wide.wait(timeout=60) == 0


def test_status_prints_position_and_estimated_start_per_queued_run(tmp_path):
    directory = tmp_path / "slots"
    directory.mkdir()
    import testqueue
    running = testqueue.RunRecord(directory, "running-run", "interactive", 100.0, 4)
    running.update(state="running", granted=2, started=time.time() - 10)
    queued = testqueue.RunRecord(directory, "queued-run", "background", 500.0, 0)
    out = subprocess.run([sys.executable, str(SCRIPT), "status"], env=_env(tmp_path, 2), capture_output=True,
                         text=True, timeout=30).stdout
    queued.close()
    running.close()
    assert any("running" in line and "running-run" in line and "2 worker" in line for line in out.splitlines()), out
    assert "Traceback" not in out
    assert any("queued" in line and "position 1" in line and "starts about" in line and "queued-run" in line
               for line in out.splitlines()), out


def test_a_live_lease_from_older_code_keeps_arrival_order_and_lifts_the_cap(tmp_path):
    module = _module()
    import testslots_policy
    now = time.time()

    def slot(name, version, estimate, klass="interactive", granted=0):
        data = {"label": name, "pid": 1, "want": 1, "minimum": 1, "granted": granted, "since": now, "version": version,
                "class": klass, "estimate": estimate, "token": name}
        return module.Slot(tmp_path / f"{name}.slot", data)

    older, newer = slot("a-old", 1, None), slot("b-new", testslots_policy.VERSION, 1.0)
    assert testslots_policy.has_legacy([older, newer])
    assert module._decide(newer.data, newer.path, [older, newer], 4)[0] == 0          # arrival order: the older lease is first
    assert module._decide(older.data, older.path, [older, newer], 4)[0] > 0
    current = slot("a-cur", testslots_policy.VERSION, 500.0)
    assert not testslots_policy.has_legacy([current, newer])
    assert module._decide(newer.data, newer.path, [current, newer], 4)[0] > 0           # the shorter request goes first
