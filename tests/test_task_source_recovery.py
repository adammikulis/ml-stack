"""Offline source recovery retains pending tasks and immutable resource history."""

from copy import deepcopy
from pathlib import Path

import pytest
from taskboard_kit import board as _board_fixture

from ml_stack import worktreerules
from ml_stack.graph.store import GraphStore
from ml_stack.workspace import localagent, task_source_recovery as recovery, task_worktree_recovery
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.task_graph import record

board = _board_fixture


@pytest.fixture
def stopped(board, monkeypatch):
    board.board.claim(board.child, board.task['id'], board.allocation['allocation_id'])
    board.board.block(board.parent, board.task['id'], 'Pending native task')
    monkeypatch.setattr(task_worktree_recovery, 'pid_exists', lambda pid: False)
    task_worktree_recovery.dematerialize(board.ws, board.parent, board.task['id'], 'Worker stopped')
    board.target = worktreerules.checkouts(board.source)[1]
    return board


def test_source_rebind_preserves_specs_grants_states_and_historical_allocations(stopped):
    kit = stopped
    before = kit.board.get(kit.parent, kit.task['id'])
    registry = deepcopy(kit.ws.registry._load())
    with GraphStore(kit.base / 'coordination.db') as graph:
        allocation = record(graph, kit.allocation['allocation_id'], 'allocation')
        scope = record(graph, 'task-worktree:' + kit.task['id'].split(':')[1], 'task-worktree')
    result = recovery.rebind(kit.ws, kit.parent, kit.worker_id, str(kit.target), 'Retire source anchor')
    after = kit.board.get(kit.parent, kit.task['id'])
    assert result['verified'] and result['tasks'] == [kit.task['id']]
    assert kit.ws.registry._load() == registry
    for key in ('state', 'blocked_reason', 'failures', 'spec_hash', 'project', 'assignees'):
        assert after[key] == before[key]
    assert after['checkpoints'][:-1] == before['checkpoints']
    assert localagent.load(kit.ws, 'native-worker').project == str(kit.target)
    with GraphStore(kit.base / 'coordination.db') as graph:
        assert record(graph, kit.allocation['allocation_id'], 'allocation') == allocation
        updated = record(graph, scope['id'], 'task-worktree')
        assert updated == {**scope, 'source_project': str(kit.target)}
    repeated = recovery.rebind(kit.ws, kit.parent, kit.worker_id, str(kit.target), 'Verify recovery')
    assert repeated['recovery'] == result['recovery']
    assert kit.board.get(kit.parent, kit.task['id'])['checkpoints'] == after['checkpoints']


@pytest.mark.redteam
@pytest.mark.parametrize('kind', ['caller', 'live', 'materialized', 'repository', 'baseline', 'creator', 'grant'])
def test_source_rebind_refuses_unrelated_or_active_assignments(stopped, monkeypatch, kind):
    kit = stopped
    target, token = kit.target, kit.parent
    if kind == 'caller':
        token = kit.child
    elif kind == 'live':
        monkeypatch.setattr(task_worktree_recovery, 'pid_exists', lambda pid: pid == 555)
        monkeypatch.setattr(task_worktree_recovery, 'started_at', lambda pid: 42)
    elif kind == 'repository':
        target = kit.source.parent / 'unrelated'
        target.mkdir()
        from ml_stack.workspace import integration_git as repo
        repo.git(target, 'init')
    else:
        with GraphStore(kit.base / 'coordination.db') as graph:
            if kind in ('baseline', 'materialized'):
                scope = record(graph, 'task-worktree:' + kit.task['id'].split(':')[1], 'task-worktree')
                if kind == 'materialized':
                    Path(scope['project']).mkdir()
                else:
                    graph.upsert_node({'id': scope['id'], 'kind': 'task-worktree',
                                       'attrs': {**scope, 'baseline_commit': 'a' * 40}})
            else:
                task = record(graph, kit.task['id'], 'task')
                if kind == 'creator':
                    task['created_by'] = 'unrelated'
                else:
                    from ml_stack.workspace.task_schema import SPEC_FIELDS, fingerprint
                    task['project'] = {'key': 'unrelated', 'name': 'other'}
                    task['spec_hash'] = fingerprint({key: task[key] for key in SPEC_FIELDS})
                graph.upsert_node({'id': task['id'], 'kind': 'task', 'attrs': task})
    before = localagent.load(kit.ws, 'native-worker')
    with pytest.raises((Denied, RuntimeError)):
        recovery.rebind(kit.ws, token, kit.worker_id, str(target), 'Offline recovery')
    assert localagent.load(kit.ws, 'native-worker') == before


def test_prepared_rebind_resumes_after_worker_file_interruption(stopped, monkeypatch):
    kit = stopped
    real_save = localagent.save
    def crash(ws, worker):
        real_save(ws, worker)
        raise SystemExit('Interrupted after worker update')
    monkeypatch.setattr(localagent, 'save', crash)
    with pytest.raises(SystemExit):
        recovery.rebind(kit.ws, kit.parent, kit.worker_id, str(kit.target), 'Resume source recovery')
    monkeypatch.setattr(localagent, 'save', real_save)
    result = recovery.rebind(kit.ws, kit.parent, kit.worker_id, str(kit.target), 'Resume source recovery')
    assert result['verified']
    with GraphStore(kit.base / 'coordination.db') as graph:
        journals = graph.nodes('source-recovery')
        assert len(journals) == 1 and journals[0]['attrs']['state'] == 'committed'


def test_failed_graph_update_restores_worker_and_resumes_prepared_journal(stopped, monkeypatch):
    kit = stopped
    original, real_save = localagent.load(kit.ws, 'native-worker'), recovery.save
    def fail(graph, kind, row):
        if kind == 'task-worktree':
            raise RuntimeError('Graph write interrupted')
        real_save(graph, kind, row)
    monkeypatch.setattr(recovery, 'save', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        recovery.rebind(kit.ws, kit.parent, kit.worker_id, str(kit.target), 'Recover source')
    assert localagent.load(kit.ws, 'native-worker') == original
    monkeypatch.setattr(recovery, 'save', real_save)
    assert recovery.rebind(kit.ws, kit.parent, kit.worker_id, str(kit.target), 'Recover source')['verified']


@pytest.mark.redteam
@pytest.mark.parametrize('drift', ['repository', 'dirty'])
def test_committed_retry_revalidates_repository_and_clean_source(stopped, monkeypatch, drift):
    kit = stopped
    recovery.rebind(kit.ws, kit.parent, kit.worker_id, str(kit.target), 'Initial recovery')
    if drift == 'dirty':
        (kit.target / 'code.py').write_text('changed = True\n')
    else:
        actual = recovery.worktreerules.checkouts
        monkeypatch.setattr(recovery.worktreerules, 'checkouts', lambda target:
                            (kit.target, kit.target.parent / 'different-primary') if Path(target) == kit.target
                            else actual(target))
    with pytest.raises(Denied):
        recovery.rebind(kit.ws, kit.parent, kit.worker_id, str(kit.target), 'Verify recovery')


@pytest.mark.redteam
def test_authenticated_revocation_waits_for_source_transaction(stopped, monkeypatch):
    import os
    import subprocess
    import sys
    import time

    kit = stopped
    ready, finished = kit.base / 'revoke-ready', kit.base / 'revoke-finished'
    processes = []
    actual = recovery._apply
    def competing(ws, journal, runner, scopes):
        code = """import os, sys
from pathlib import Path
from ml_stack.workspace.service import Workspace
ws = Workspace(Path(sys.argv[1]))
Path(sys.argv[3]).write_text('ready')
ws.revoke(os.environ['ML_STACK_TEST_RECOVERY_PARENT'], sys.argv[2])
Path(sys.argv[4]).write_text('finished')
"""
        process = subprocess.Popen([sys.executable, '-c', code, str(kit.base), kit.worker_id,
                                    str(ready), str(finished)],
                                   env={**os.environ, 'ML_STACK_TEST_RECOVERY_PARENT': kit.parent})
        processes.append(process)
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists() and process.poll() is None
        time.sleep(0.05)
        assert not finished.exists()
        actual(ws, journal, runner, scopes)
        assert not finished.exists()
    monkeypatch.setattr(recovery, '_apply', competing)
    try:
        result = recovery.rebind(kit.ws, kit.parent, kit.worker_id, str(kit.target), 'Authorized recovery')
        assert result['verified']
        assert processes[0].wait(timeout=10) == 0 and finished.exists()
        with pytest.raises(Denied):
            recovery.rebind(kit.ws, kit.parent, kit.worker_id, str(kit.target), 'Revoked worker')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
