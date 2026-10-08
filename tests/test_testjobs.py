"""Background test jobs: the job table, the detached runner's lifecycle, ownership, cancel and pruning."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import testjobs  # noqa: E402
import testreuse_store as storage  # noqa: E402

pytestmark = pytest.mark.slow

RUNNER = f'''
import os, signal, sys, time
from pathlib import Path
sys.path.insert(0, {str(SCRIPTS)!r})
sys.path.insert(0, {str(SCRIPTS.parent / "src")!r})
import testjobs
import testreuse_store as storage

jobs = testjobs.Jobs(Path(os.environ["JOBS_HOME"]))
job = sys.argv[2]
spec = jobs.read(job, "spec.json")
me = {{"pid": os.getpid(), "started": storage.process_start(os.getpid())}}
jobs.write(job, "status.json", {{"state": "running", "at": time.time(), **me}})
signal.signal(signal.SIGTERM, lambda *_: (jobs.write(job, "status.json", {{"state": "cancelled", "exit": 130,
    "finished": time.time(), **me}}), os._exit(130)))
print("working", flush=True)
time.sleep(float(spec["argv"][1]))
jobs.write(job, "status.json", {{"state": "done", "exit": int(spec["argv"][2]), "finished": time.time(),
    "summary": {{"ran": 1, "reused": 2}}, **me}})
'''

ALICE = {"id": "alice"}
BOB = {"id": "bob"}


@pytest.fixture
def table(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_HOME", str(tmp_path / "store"))
    script = tmp_path / "runner.py"
    script.write_text(RUNNER)
    (tmp_path / "tests").mkdir()
    return testjobs.Jobs(tmp_path / "store"), script, tmp_path


def start(table, argv, owner=ALICE):
    jobs, script, root = table
    return jobs.submit(argv, owner, root, script, {})


def test_submit_returns_at_once_and_the_job_survives_to_completion(table):
    jobs, _, root = table
    began = time.monotonic()
    job = start(table, ["tests/x.py", "1.5", "0"])
    assert time.monotonic() - began < 1.0 and jobs.state(job)["state"] in ("queued", "running")
    done = jobs.wait(job, 30)
    assert done["state"] == "done" and done["exit"] == 0 and done["summary"] == {"ran": 1, "reused": 2}
    assert "working" in (jobs.path(job) / "log").read_text()
    assert jobs.result(job, root)["failing"] == []


def test_cancel_stops_a_running_job_for_its_owner_only(table):
    jobs, _, _ = table
    job = start(table, ["tests/x.py", "60", "0"])
    deadline = time.monotonic() + 20
    while jobs.state(job).get("state") != "running" and time.monotonic() < deadline:
        time.sleep(0.1)
    time.sleep(0.5)
    with pytest.raises(testjobs.Refused, match="belongs to alice"):
        jobs.cancel(job, "bob")
    assert jobs.cancel(job, "alice").endswith("cancelled")
    assert jobs.wait(job, 20)["state"] == "cancelled"
    assert "already cancelled" in jobs.cancel(job, "alice")


def test_only_one_whole_tier_job_runs_at_a_time(table):
    jobs, _, root = table
    first = start(table, ["full", "30", "0"])
    with pytest.raises(testjobs.Refused, match=f"job {first} \\(alice\\) already runs a whole tier"):
        start(table, ["all", "1", "0"], BOB)
    explicit = start(table, ["tests/x.py", "0.1", "0"], BOB)
    assert testjobs.is_full(["full"], root) and not testjobs.is_full(["all", "tests"], root)
    jobs.cancel(first, "alice")
    assert jobs.wait(explicit, 20)["state"] == "done"
    assert jobs.wait(first, 20)["state"] == "cancelled"
    assert start(table, ["full", "0.1", "0"], BOB)


def test_a_job_whose_runner_died_is_reported_failed(table):
    jobs, _, _ = table
    job = start(table, ["tests/x.py", "60", "0"])
    time.sleep(1.0)
    status = jobs.read(job)
    os.kill(int(status["pid"]), 9)
    deadline = time.monotonic() + 20
    while jobs.state(job)["state"] != "failed" and time.monotonic() < deadline:
        time.sleep(0.2)
    assert jobs.state(job)["state"] == "failed"
    assert jobs.live_full(table[2]) == []


def test_failing_node_ids_come_from_the_kept_junit(table):
    jobs, _, root = table
    job = start(table, ["tests/x.py", "0.1", "1"])
    jobs.wait(job, 20)
    (jobs.path(job) / "junit-1.xml").write_text(
        '<testsuites><testsuite><testcase classname="tests.test_x" name="test_a"><failure/></testcase>'
        '<testcase classname="tests.test_x" name="test_b"/></testsuite></testsuites>')
    result = jobs.result(job, root)
    assert result["exit"] == 1 and result["failing"] == ["tests/test_x.py::test_a"]
    assert result["junit"] == [str(jobs.path(job) / "junit-1.xml")]


def test_old_finished_jobs_are_pruned_and_live_ones_kept(table):
    jobs, _, _ = table
    old = start(table, ["tests/x.py", "0.1", "0"])
    jobs.wait(old, 20)
    status = jobs.read(old)
    status["finished"] = time.time() - testjobs.KEEP_S - 10
    jobs.write(old, "status.json", status)
    live = start(table, ["tests/x.py", "20", "0"], BOB)
    jobs.prune()
    assert old not in jobs.ids() and live in jobs.ids()
    jobs.cancel(live, "bob")


def test_job_ids_cannot_escape_the_job_table(table):
    jobs, _, _ = table
    with pytest.raises(testjobs.Refused):
        jobs.path("../../etc")
    assert storage.alive(os.getpid(), storage.process_start(os.getpid()))
