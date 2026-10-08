"""scripts/testhistory.py, testclass.py, testhours.py, testqueue.py: duration history, classification and queue forecast."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import testclass  # noqa: E402
import testhistory  # noqa: E402
import testhours  # noqa: E402
import testqueue  # noqa: E402


def _tree(tmp_path, files):
    for name in files:
        path = tmp_path / "tests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_x():\n    pass\n")
    return tmp_path


def _history(per_file, tests_each=1):
    """tests recorded as {file: seconds}: one entry per file (cpu seconds)."""
    out = {}
    for file, seconds in per_file.items():
        for n in range(tests_each):
            out[f"tests/{file}::test_{n}"] = {"cpu": seconds / tests_each, "wall": 0.0, "n": 3}
    return out


def test_a_sample_is_recorded_and_merged_into_one_store(tmp_path):
    root = _tree(tmp_path, ["test_a.py"])
    store = tmp_path / "hist" / "durations.json"
    testhistory.merge(store, root, {"tests/test_a.py::test_x": (2.0, 3.0)})
    testhistory.merge(store, root, {"tests/test_a.py::test_x": (4.0, 3.0)})
    entry = testhistory.load(store)["tests/test_a.py::test_x"]
    assert entry["n"] == 2 and 2.0 < entry["cpu"] < 4.0
    assert testhistory.seconds({"cpu": 0.1, "wall": 8.0}) == 2.0      # a quarter of the wall time is the floor
    assert testhistory.seconds({"cpu": 5.0, "wall": 8.0}) == 5.0


def test_one_loaded_outlier_barely_moves_the_estimate(tmp_path):
    entry = None
    for _ in range(5):
        entry = testhistory.update(entry, 1.0, 1.0)
    entry = testhistory.update(entry, 500.0, 500.0)
    assert entry["cpu"] < 2.0 and entry["n"] == 6


WRITER = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import testhistory
store, root, tag = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
for n in range(15):
    testhistory.merge(store, root, {f"tests/test_a.py::test_{tag}{n}": (1.0, 1.0)})
'''


def test_two_concurrent_writers_lose_nothing(tmp_path):
    root = _tree(tmp_path, ["test_a.py"])
    store = tmp_path / "hist" / "durations.json"
    writers = [subprocess.Popen([sys.executable, "-c", WRITER, str(SCRIPTS), str(store), str(root), tag],
                                env={**os.environ, "PYTHONPATH": str(ROOT / "src")}) for tag in ("a", "b")]
    assert all(w.wait(timeout=120) == 0 for w in writers)
    assert len(testhistory.load(store)) == 30
    json.loads(store.read_text())


def test_tests_that_no_longer_exist_are_dropped(tmp_path):
    root = _tree(tmp_path, ["test_a.py", "test_b.py"])
    store = tmp_path / "durations.json"
    testhistory.merge(store, root, {"tests/test_a.py::test_old": (1, 1), "tests/test_a.py::test_new": (1, 1),
                                    "tests/test_b.py::test_x": (1, 1), "tests/test_gone.py::test_x": (1, 1)})
    testhistory.merge(store, root, {}, {"tests/test_a.py": {"tests/test_a.py::test_new"}})
    assert set(testhistory.load(store)) == {"tests/test_a.py::test_new", "tests/test_b.py::test_x"}


def test_the_pytest_plugin_records_passing_tests_only(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "test_p.py").write_text("def test_pass():\n    sum(range(10**6))\n\ndef test_fail():\n    assert False\n")
    store = tmp_path / "hist" / "durations.json"
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "testdurations", "-p", "no:cacheprovider",
                           "-p", "no:xdist", "test_p.py"], cwd=case, capture_output=True, text=True, timeout=120,
                          env={**os.environ, "DEV_TEST_HISTORY": str(store), "PYTHONPATH": str(SCRIPTS),
                               "PYTEST_ADDOPTS": ""})
    assert done.returncode == 1, done.stdout
    assert set(testhistory.load(store)) == {"test_p.py::test_pass"}


def test_unknown_files_count_thirty_seconds_each_and_selectors_sum_per_file(tmp_path):
    root = _tree(tmp_path, ["test_a.py", "test_b.py", "test_new.py"])
    tests = _history({"test_a.py": 10.0, "test_b.py": 20.0})
    assert testhistory.estimate(tests, root, ["tests/test_a.py"]).seconds == 10.0
    assert testhistory.estimate(tests, root, ["tests/test_new.py"]).seconds == testhistory.DEFAULT_FILE_S
    everything = testhistory.estimate(tests, root, [])
    assert everything.seconds == 10.0 + 20.0 + testhistory.DEFAULT_FILE_S and everything.unknown == 1
    assert testhistory.estimate(tests, root, ["tests/test_a.py::test_0"]).seconds == 10.0
    assert everything.wall(3) == everything.seconds / 3


def _verdict(root, tests, tier="all", rest=(), workers=1, **flags):
    return testclass.classify(tier, list(rest), root, testclass.Basis(tests, workers), flags.get("background", False))


def test_the_estimate_decides_the_class_at_the_threshold(tmp_path):
    root = _tree(tmp_path, [f"test_{n}.py" for n in range(60)] + ["test_edge.py"])
    base = _history({f"test_{n}.py": 0.01 for n in range(60)})
    edge = testhistory.THRESHOLD_S
    below = {**base, "tests/test_edge.py::test_0": {"cpu": edge - 0.5 - 0.6, "wall": 0.0, "n": 3}}
    at = {**base, "tests/test_edge.py::test_0": {"cpu": edge - 0.6, "wall": 0.0, "n": 3}}
    assert _verdict(root, below, rest=["tests/test_edge.py"]).klass == "interactive"
    assert _verdict(root, at, rest=["tests/test_edge.py"]).klass == "interactive"
    over = {**base, "tests/test_edge.py::test_0": {"cpu": edge, "wall": 0.0, "n": 3}}
    assert _verdict(root, over, rest=["tests/test_edge.py"]).klass == "background"


def test_class_flips_with_the_estimate_whatever_the_tier(tmp_path):
    root = _tree(tmp_path, [f"test_{n}.py" for n in range(60)])
    fast = _history({f"test_{n}.py": 1.0 for n in range(60)})
    slow = _history({f"test_{n}.py": 20.0 for n in range(60)})
    assert _verdict(root, fast, tier="full", workers=4).klass == "interactive"      # 60 s of work on 4 workers
    assert _verdict(root, slow, tier="full", workers=4).klass == "background"       # 300 s on 4 workers
    assert _verdict(root, slow, tier="all", rest=["tests/test_0.py"]).klass == "interactive"
    many = _tree(tmp_path / "many", [f"test_{n}.py" for n in range(200)])
    verdict = _verdict(many, _history({f"test_{n}.py": 1.0 for n in range(200)}), workers=1)
    assert verdict.klass == "background" and verdict.estimate_s == 200.0
    assert "estimated" in testclass.notice(verdict)
    assert testclass.estimate_line(verdict).startswith("test: estimated 3 min from 200 recorded tests")


def test_without_enough_history_the_command_shape_decides(tmp_path):
    root = _tree(tmp_path, ["test_a.py"])
    assert _verdict(root, {}, tier="full").klass == "background"
    assert _verdict(root, {}, tier="slow").klass == "background"
    assert _verdict(root, {}, tier="all").klass == "background"
    assert _verdict(root, {}, tier="fast").klass == "background"
    assert _verdict(root, {}, tier="all", rest=["tests/test_a.py"]).klass == "interactive"
    assert _verdict(root, {}, tier="all", rest=["tests/test_a.py::test_x"]).klass == "interactive"
    assert _verdict(root, {}, tier="all", rest=["tests/test_a.py", "--redteam"]).klass == "background"
    assert _verdict(root, {}, tier="all", rest=["-k", "x"]).klass == "background"
    assert _verdict(root, {}, tier="quick").klass == "interactive"
    assert _verdict(root, {}, tier="gate").klass == "interactive"
    assert _verdict(root, {}, tier="record").klass == "background"
    assert _verdict(root, {}, tier="all", rest=["tests/test_a.py"], background=True).klass == "background"


def test_scripts_test_takes_background_and_land_passes_it():
    assert '"--background"' in (SCRIPTS / "test").read_text()
    assert '"--background"' in (SCRIPTS / "land_check.py").read_text()
    assert "--now" not in (SCRIPTS / "test").read_text()


def _at(day, hour, minute=0):
    return dt.datetime(2026, 10, day, hour, minute)          # 2026-10-05 is a Monday


@pytest.mark.parametrize(("when", "capped"), [
    (_at(5, 7, 59), False), (_at(5, 8, 0), True), (_at(5, 20, 59), True), (_at(5, 21, 0), False),
    (_at(10, 12), False), (_at(11, 12), False), (_at(9, 12), True),
])
def test_the_cap_applies_only_inside_weekday_normal_hours(when, capped):
    assert testhours.cap_applies(when, testhours.parse_window(None)) is capped


def test_the_hours_setting_parses_and_off_keeps_the_cap_on():
    assert testhours.parse_window("09:30-17:00") == (dt.time(9, 30), dt.time(17, 0))
    assert testhours.cap_applies(_at(5, 9, 29), testhours.parse_window("09:30-17:00")) is False
    assert testhours.parse_window("off") is None
    assert testhours.cap_applies(_at(10, 3), None) is True
    assert testhours.capped_now({"DEV_TEST_NORMAL_HOURS": "10:00-11:00"}, _at(5, 12)) is False
    assert testhours.capped_now({"DEV_TEST_NORMAL_HOURS": "nonsense"}, _at(10, 3)) is True
    for bad in ("9-17", "17:00-09:00", "25:00-26:00"):
        with pytest.raises(ValueError):
            testhours.parse_window(bad)


def _run(pid, state, klass, estimate, enqueued, **fields):
    return {"id": pid, "pid": pid, "label": f"run{pid}", "class": klass, "state": state, "estimate_s": estimate,
            "want": 0, "granted": 0, "enqueued": enqueued, "started": None, **fields}


def test_forecast_orders_short_work_first_and_projects_the_start():
    now = 1000.0
    runs = [_run(1, "running", "background", 400.0, 900.0, granted=4, started=900.0, want=4),
            _run(2, "queued", "background", 4000.0, 990.0, want=4),
            _run(3, "queued", "interactive", 40.0, 995.0, want=4)]
    views = testqueue.forecast(runs, 4, now)
    assert views[3].position == 1 and views[2].position == 2           # the short run arrived later and goes first
    assert views[3].ahead_count == 0 and views[3].ahead_estimate_s == 300.0       # only the running run's remainder
    assert views[3].start_at == now + 300.0                              # when the running run is estimated to finish
    assert views[2].ahead_count == 1 and views[2].ahead_estimate_s == 340.0
    assert views[2].start_at == pytest.approx(now + 300.0 + 40.0 / 4)
    assert views[1].state == "running"


def test_forecast_puts_an_aged_run_first(monkeypatch):
    monkeypatch.setenv("DEV_TEST_BACKGROUND_WAIT_S", "100")
    now = 1000.0
    runs = [_run(2, "queued", "background", 4000.0, 500.0), _run(3, "queued", "interactive", 40.0, 990.0)]
    views = testqueue.forecast(runs, 4, now)
    assert views[2].position == 1 and views[3].position == 2


def test_files_the_reuse_store_would_serve_cost_nothing_in_the_estimate(tmp_path):
    root = _tree(tmp_path, ["test_a.py", "test_b.py"])
    tests = _history({"test_a.py": 100.0, "test_b.py": 20.0})
    assert testhistory.estimate(tests, root, [], frozenset({"tests/test_a.py"})).seconds == 20.0
    basis = testclass.Basis({**tests, **_history({f"test_{n}.py": 0.0 for n in range(60)})}, 1,
                            frozenset({"tests/test_a.py"}))
    assert testclass.classify("all", [], root, basis).estimate_s == 20.0
