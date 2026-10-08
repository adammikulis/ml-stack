"""Durable Fleet recovery preserves execution ownership and avoids replay."""

import json
import os
import sys
import time

import pytest

from ml_stack.fleet.job_records import process_started
from ml_stack.fleet.jobs import DaemonError, Job, JobRunner


def wait_for(check, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.02)
    pytest.fail('Fleet job did not reach the expected durable state')


def command(marker, release=None):
    code = "from pathlib import Path; import os,time; p=Path(os.environ['MARKER']); p.open('a').write(str(os.getpid())+'\\n')"
    if release:
        code += "; release=Path(os.environ['RELEASE']); exec('while not release.exists(): time.sleep(0.02)')"
    return [sys.executable, '-c', code], {'MARKER': str(marker), **({'RELEASE': str(release)} if release else {})}


def test_queued_jobs_and_terminal_history_restore_without_replaying(tmp_path):
    marker = tmp_path / 'executions'
    argv, env = command(marker)
    runner = JobRunner(tmp_path / 'runner', gate=lambda: (False, 'Held'))
    job = runner.submit('queued', argv, str(tmp_path), env)
    checkpoint = runner.checkpoint_restart()
    assert checkpoint == {'queued': 1, 'running': 0}
    runner.shutdown()
    restored = JobRunner(tmp_path / 'runner')
    try:
        wait_for(lambda: restored.jobs[job.id].state == 'done')
        assert len(marker.read_text().splitlines()) == 1
    finally:
        restored.shutdown()
    history = JobRunner(tmp_path / 'runner')
    try:
        assert history.jobs[job.id].state == 'done'
        assert len(marker.read_text().splitlines()) == 1
    finally:
        history.shutdown()


def test_live_child_survives_restart_and_holds_capacity_without_duplicate_execution(tmp_path):
    marker, release = tmp_path / 'executions', tmp_path / 'release'
    argv, env = command(marker, release)
    runner = JobRunner(tmp_path / 'runner')
    job = runner.submit('running', argv, str(tmp_path), env)
    wait_for(lambda: job.state == 'running' and marker.exists())
    queued = runner.submit('later', *command(tmp_path / 'later')[:1], cwd=str(tmp_path), env=command(tmp_path / 'later')[1])
    pid = job.pid
    assert runner.checkpoint_restart() == {'queued': 1, 'running': 1}
    runner.shutdown()
    restored = JobRunner(tmp_path / 'runner')
    try:
        assert restored.jobs[job.id].pid == pid and restored.status()['busy']
        assert restored.status()['free'] == 0 and restored.jobs[queued.id].state == 'queued'
        assert process_started(pid) == job.process_started
        assert len(marker.read_text().splitlines()) == 1
        release.touch()
        wait_for(lambda: restored.jobs[queued.id].state == 'done')
        assert len(marker.read_text().splitlines()) == 1
    finally:
        release.touch()
        restored.shutdown()
        wait_for(lambda: job.state in ('done', 'failed'))


def test_queued_cancellation_is_durable(tmp_path):
    runner = JobRunner(tmp_path / 'runner', gate=lambda: (False, 'Held'))
    job = runner.submit('cancelled', [sys.executable, '-c', 'raise SystemExit(0)'], str(tmp_path))
    runner.stop(job.id)
    runner.shutdown()
    restored = JobRunner(tmp_path / 'runner')
    try:
        assert restored.jobs[job.id].state == 'stopped' and restored.status()['queued'] == 0
    finally:
        restored.shutdown()


@pytest.mark.redteam
@pytest.mark.parametrize('state', ['queued-legacy', 'launching', 'running'])
def test_uncertain_or_reused_process_records_are_never_replayed_or_signalled(tmp_path, monkeypatch, state):
    runner = JobRunner(tmp_path / 'runner', gate=lambda: (False, 'Held'))
    job = Job('uncertain', 'Uncertain work', [sys.executable, '-c', 'raise SystemExit(99)'], str(tmp_path),
              state='queued' if state == 'queued-legacy' else state,
              pid=os.getpid() if state == 'running' else None,
              process_started=(process_started(os.getpid()) or 0) + 1)
    runner.record(job)
    path = runner.job_dir(job.id) / 'job.json'
    if state == 'queued-legacy':
        document = json.loads(path.read_text())
        document.pop('version')
        path.write_text(json.dumps(document))
    runner.shutdown()
    monkeypatch.setattr('ml_stack.fleet.jobs.stop_pid', lambda pid: pytest.fail('signalled unrelated process'))
    restored = JobRunner(tmp_path / 'runner')
    try:
        assert restored.jobs[job.id].state == 'interrupted'
        assert restored.status()['queued'] == 0
        if state == 'running':
            assert restored.stop(job.id).state == 'interrupted'
        else:
            assert restored.status()['busy']
            with pytest.raises(DaemonError, match='verified process recovery'):
                restored.stop(job.id)
    finally:
        restored.shutdown()


@pytest.mark.redteam
def test_two_runners_cannot_consume_the_same_queue(tmp_path):
    runner = JobRunner(tmp_path / 'runner')
    try:
        with pytest.raises(ValueError, match='another Fleet runner'):
            JobRunner(tmp_path / 'runner')
    finally:
        runner.shutdown()


@pytest.mark.redteam
def test_symlink_record_and_failed_checkpoint_preserve_existing_runner(tmp_path, monkeypatch):
    runner = JobRunner(tmp_path / 'runner', gate=lambda: (False, 'Held'))
    job = runner.submit('queued', [sys.executable, '-c', 'raise SystemExit(0)'], str(tmp_path))
    original = runner.record
    monkeypatch.setattr(runner, 'record', lambda row: (_ for _ in ()).throw(OSError('Disk unavailable')))
    with pytest.raises(OSError, match='unavailable'):
        runner.checkpoint_restart()
    assert not runner._stop.is_set()
    monkeypatch.setattr(runner, 'record', original)
    path = runner.job_dir(job.id) / 'job.json'
    path.unlink()
    path.symlink_to(tmp_path / 'external')
    with pytest.raises(ValueError, match='symlinks'):
        runner.record(job)
    runner.shutdown()


def test_recovery_after_daemon_process_exit_keeps_child_and_its_recorded_outcome(tmp_path):
    import subprocess

    marker, release = tmp_path / 'executions', tmp_path / 'release'
    argv, env = command(marker, release)
    owner = """import json, os, sys, time
from pathlib import Path
from ml_stack.fleet.jobs import JobRunner
root, cwd, argv, env = json.loads(sys.argv[1])
runner = JobRunner(Path(root))
job = runner.submit('owned child', argv, cwd, env)
while job.state != 'running': time.sleep(0.01)
runner.checkpoint_restart()
print(json.dumps({'id': job.id, 'pid': job.pid}), flush=True)
os._exit(0)
"""
    result = subprocess.run([sys.executable, '-c', owner,
                             json.dumps([str(tmp_path / 'runner'), str(tmp_path), argv, env])],
                            capture_output=True, text=True, check=True, timeout=10)
    announced = json.loads(result.stdout)
    restored = JobRunner(tmp_path / 'runner')
    try:
        assert restored.jobs[announced['id']].state == 'running'
        assert restored.jobs[announced['id']].pid == announced['pid']
        wait_for(marker.exists)
        assert len(marker.read_text().splitlines()) == 1
        release.touch()
        wait_for(lambda: restored.snapshot()[0]['state'] == 'done')
        assert restored.jobs[announced['id']].returncode == 0
        assert len(marker.read_text().splitlines()) == 1
    finally:
        release.touch()
        restored.shutdown()


def test_closed_runner_completion_cannot_overwrite_new_owner_terminal_state(tmp_path):
    marker, release = tmp_path / 'executions', tmp_path / 'release'
    argv, env = command(marker, release)
    original = JobRunner(tmp_path / 'runner')
    job = original.submit('running', argv, str(tmp_path), env)
    wait_for(lambda: job.state == 'running' and marker.exists())
    original.checkpoint_restart()
    original.shutdown()
    restored = JobRunner(tmp_path / 'runner')
    try:
        active = restored.jobs[job.id]
        active.state = 'stopped'
        active.finished_at = time.time()
        restored.record(active)
        release.touch()
        wait_for(lambda: job.state == 'done')
        saved = json.loads((restored.job_dir(job.id) / 'job.json').read_text())
        assert saved['state'] == 'stopped' and restored.snapshot()[0]['state'] == 'stopped'
    finally:
        release.touch()
        restored.shutdown()


def test_failed_restart_checkpoint_restores_dequeued_state_and_queue(tmp_path, monkeypatch):
    runner = JobRunner(tmp_path / 'runner', gate=lambda: (False, 'Held'))
    job = runner.submit('reserved', [sys.executable, '-c', 'raise SystemExit(0)'], str(tmp_path))
    with runner._lock:
        runner._queue.remove(job.id)
        runner._starting.add(job.id)
        job.state = 'launching'
    def fail(row):
        raise OSError('Checkpoint unavailable')
    monkeypatch.setattr(runner, 'record', fail)
    try:
        with pytest.raises(OSError):
            runner.checkpoint_restart()
        assert job.state == 'launching' and runner._queue == [] and not runner._stop.is_set()
    finally:
        runner.shutdown()


@pytest.mark.redteam
@pytest.mark.parametrize('field,value', [('pid', True), ('pid', -2), ('process_started', 'wrong'),
                                         ('submitted_at', float('nan')), ('version', 3)])
def test_malformed_ownership_record_cannot_resume_execution(tmp_path, field, value):
    runner = JobRunner(tmp_path / 'runner', gate=lambda: (False, 'Held'))
    job = runner.submit('queued', [sys.executable, '-c', 'raise SystemExit(0)'], str(tmp_path))
    path = runner.job_dir(job.id) / 'job.json'
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    runner.shutdown()
    with pytest.raises(ValueError):
        JobRunner(tmp_path / 'runner')


def test_running_checkpoint_failure_keeps_child_capacity_and_freezes_queue(tmp_path, monkeypatch):
    marker, release, later = tmp_path / 'executions', tmp_path / 'release', tmp_path / 'later'
    runner = JobRunner(tmp_path / 'runner', gate=lambda: (False, 'Held'))
    argv, env = command(marker, release)
    job = runner.submit('running', argv, str(tmp_path), env)
    next_argv, next_env = command(later)
    runner.submit('next', next_argv, str(tmp_path), next_env)
    actual = runner.record
    failed = []
    def fail_running(row):
        if row.id == job.id and row.state == 'running' and not failed:
            failed.append(True)
            raise OSError('Running checkpoint unavailable')
        actual(row)
    monkeypatch.setattr(runner, 'record', fail_running)
    runner.gate = None
    runner._wake.set()
    try:
        wait_for(lambda: bool(failed) and marker.exists())
        assert runner.status()['busy'] and runner._stop.is_set()
        assert job.id in runner._running and not later.exists()
        release.touch()
        wait_for(lambda: job.state == 'done')
        assert not later.exists()
    finally:
        release.touch()
        runner.shutdown()


def test_uncertain_launch_reserves_capacity_across_restarts(tmp_path):
    marker = tmp_path / 'must-not-run'
    runner = JobRunner(tmp_path / 'runner', gate=lambda: (False, 'Held'))
    uncertain = Job('uncertain-launch', 'Prior launch', [sys.executable, '-c', 'pass'], str(tmp_path), state='launching')
    runner.record(uncertain)
    argv, env = command(marker)
    runner.submit('queued', argv, str(tmp_path), env)
    runner.shutdown()
    for _ in range(2):
        restored = JobRunner(tmp_path / 'runner')
        try:
            assert restored.jobs[uncertain.id].state == 'interrupted'
            assert restored.status()['busy'] and restored.status()['free'] == 0
            assert restored.status()['queued'] == 1 and not marker.exists()
        finally:
            restored.shutdown()


@pytest.mark.redteam
def test_inaccessible_process_identity_holds_capacity_without_signalling(tmp_path, monkeypatch):
    from ml_stack.fleet import job_records
    runner = JobRunner(tmp_path / 'runner', gate=lambda: (False, 'Held'))
    job = Job('unverifiable', 'Unverifiable process', [sys.executable, '-c', 'pass'], str(tmp_path),
              state='running', pid=os.getpid(), process_started=process_started(os.getpid()))
    runner.record(job)
    runner.shutdown()
    monkeypatch.setattr(job_records, 'ownership', lambda job: None)
    monkeypatch.setattr('ml_stack.fleet.jobs.ownership', lambda job: None)
    monkeypatch.setattr('ml_stack.fleet.jobs.stop_pid', lambda pid: pytest.fail('signalled inaccessible process'))
    restored = JobRunner(tmp_path / 'runner')
    try:
        assert restored.jobs[job.id].state == 'interrupted'
        assert restored.status()['busy'] and restored.status()['free'] == 0
        with pytest.raises(DaemonError):
            restored.stop(job.id)
    finally:
        restored.shutdown()


def test_frozen_and_closed_runners_refuse_new_submissions(tmp_path):
    runner = JobRunner(tmp_path / 'runner', gate=lambda: (False, 'Held'))
    runner.checkpoint_restart()
    with pytest.raises(DaemonError, match='frozen'):
        runner.submit('late', [sys.executable, '-c', 'pass'], str(tmp_path))
    runner.shutdown()
    with pytest.raises(DaemonError, match='frozen'):
        runner.submit('late', [sys.executable, '-c', 'pass'], str(tmp_path))
    assert runner.jobs == {}


def test_record_write_finishes_before_generation_lock_release(tmp_path, monkeypatch):
    import threading

    from ml_stack.fleet import job_records

    runner = JobRunner(tmp_path / 'runner', gate=lambda: (False, 'Held'))
    job = runner.submit('history', [sys.executable, '-c', 'pass'], str(tmp_path))
    job.state = 'done'
    entered, resume, closed = threading.Event(), threading.Event(), threading.Event()
    actual, calls = job_records.write_json, []
    def paused(path, row):
        if not calls:
            calls.append(True)
            entered.set()
            assert resume.wait(10)
        actual(path, row)
    monkeypatch.setattr(job_records, 'write_json', paused)
    writer = threading.Thread(target=lambda: runner.record(job))
    shutdown = threading.Thread(target=lambda: (runner.shutdown(), closed.set()))
    writer.start()
    assert entered.wait(10)
    shutdown.start()
    try:
        assert not closed.wait(0.05)
        with pytest.raises(ValueError, match='another Fleet runner'):
            JobRunner(tmp_path / 'runner')
    finally:
        resume.set()
        writer.join(10)
        shutdown.join(10)
    assert closed.is_set() and not writer.is_alive() and not shutdown.is_alive()
    restored = JobRunner(tmp_path / 'runner')
    try:
        assert restored.jobs[job.id].state == 'done'
    finally:
        restored.shutdown()
