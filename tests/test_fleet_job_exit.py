"""A Fleet job that finishes while its daemon restarts keeps its own exit code."""

import json
import sys
import time

import pytest

from ml_stack.fleet import job_exit
from ml_stack.fleet.job_records import process_started
from ml_stack.fleet.jobs import Job, JobRunner


def wait_for(check, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.02)
    pytest.fail('Fleet job did not reach the expected state')


def waiting(release, code):
    program = (f"import os,time\nwhile not os.path.exists({str(release)!r}): time.sleep(0.02)\n"
               f"raise SystemExit({code})")
    return [sys.executable, '-c', program]


@pytest.mark.parametrize('code', [0, 3])
def test_job_finishing_during_a_restart_keeps_its_exit_code(tmp_path, code):
    release = tmp_path / 'release'
    runner = JobRunner(tmp_path / 'runner')
    job = runner.submit('finishing', waiting(release, code), str(tmp_path))
    wait_for(lambda: job.state == 'running' and job.pid)
    pid = job.pid
    assert runner.checkpoint_restart() == {'queued': 0, 'running': 1}
    runner.shutdown()
    release.touch()
    wait_for(lambda: process_started(pid) is None)
    restored = JobRunner(tmp_path / 'runner')
    try:
        row = restored.jobs[job.id]
        assert (row.state, row.returncode) == (('done' if code == 0 else 'failed'), code)
        assert restored.status()['free'] == restored.slots
    finally:
        restored.shutdown()


def test_job_finishing_after_a_restart_is_settled_with_its_exit_code(tmp_path):
    release = tmp_path / 'release'
    runner = JobRunner(tmp_path / 'runner')
    job = runner.submit('finishing', waiting(release, 7), str(tmp_path))
    wait_for(lambda: job.state == 'running' and job.pid)
    runner.checkpoint_restart()
    runner.shutdown()
    restored = JobRunner(tmp_path / 'runner')
    try:
        assert restored.jobs[job.id].state == 'running'
        release.touch()
        wait_for(lambda: restored.snapshot() and restored.jobs[job.id].state == 'failed')
        assert restored.jobs[job.id].returncode == 7
    finally:
        restored.shutdown()


def test_stopping_a_job_reaches_its_command_through_the_exit_recorder(tmp_path):
    runner = JobRunner(tmp_path / 'runner')
    job = runner.submit('long', [sys.executable, '-c', 'import time; time.sleep(60)'], str(tmp_path))
    try:
        wait_for(lambda: job.state == 'running' and job.pid)
        pid = job.pid
        runner.stop(job.id, grace_s=5)
        wait_for(lambda: process_started(pid) is None)
        assert runner.jobs[job.id].state == 'stopped'
    finally:
        runner.shutdown()


@pytest.mark.redteam
def test_exit_record_for_another_process_is_not_trusted(tmp_path):
    (tmp_path / job_exit.NAME).write_text(json.dumps({'pid': 1, 'returncode': 0}))
    assert job_exit.read(tmp_path, 2) is None
    assert job_exit.read(tmp_path, 1) == 0
    (tmp_path / job_exit.NAME).write_text(json.dumps({'pid': 1, 'returncode': True}))
    assert job_exit.read(tmp_path, 1) is None


def test_a_launch_in_progress_looks_queued_to_clients_but_is_recorded_exactly(tmp_path):
    runner = JobRunner(tmp_path / 'runner', gate=lambda: (False, 'Held'))
    try:
        job = Job('launching', 'Launching', [sys.executable, '-c', 'pass'], str(tmp_path), state='launching')
        runner.record(job)
        assert job.public()['state'] == 'queued'
        saved = json.loads((runner.job_dir(job.id) / 'job.json').read_text())
        assert saved['state'] == 'launching'
    finally:
        runner.shutdown()
