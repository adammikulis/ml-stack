"""Inactive unchanged task recovery preserves pending graph history."""

from pathlib import Path

import pytest
from taskboard_kit import board as _board_fixture

from poolhouse.graph.store import GraphStore
from poolhouse.workspace import task_worktree_recovery as recovery, task_worktrees
from poolhouse.workspace.identity import Denied
from poolhouse.workspace.task_graph import record

board = _board_fixture


@pytest.fixture
def inactive(board, monkeypatch):
    board.board.claim(board.child, board.task['id'], board.allocation['allocation_id'])
    board.board.block(board.parent, board.task['id'], 'Native turn cancelled')
    monkeypatch.setattr(recovery, 'pid_exists', lambda pid: False)
    return board


def test_dematerialization_preserves_blockage_and_checkpoints_and_can_rematerialize(inactive):
    kit = inactive
    before = kit.board.get(kit.parent, kit.task['id'])
    result = recovery.dematerialize(kit.ws, kit.parent, kit.task['id'], 'Stopped worker verified')
    after = kit.board.get(kit.parent, kit.task['id'])
    assert result['cleanup_verified'] and result['state'] == 'blocked'
    assert after['blocked_reason'] == before['blocked_reason']
    assert after['failures'] == before['failures']
    assert after['checkpoints'][:-1] == before['checkpoints']
    with GraphStore(kit.base / 'coordination.db') as graph:
        scope = record(graph, 'task-worktree:' + kit.task['id'].split(':')[1], 'task-worktree')
    assert scope['state'] == 'reserved' and not Path(scope['project']).exists()
    assert kit.ws.who_owns('worktree', scope['project']) is None
    kit.board.resume(kit.parent, kit.task['id'], 'Ready for reassignment')
    task_worktrees.prepare(kit.ws, kit.parent, kit.worker_id, kit.task['id'])
    assert kit.ws.who_owns('worktree', scope['project'])['owner'] == 'lead'
    with GraphStore(kit.base / 'coordination.db') as graph:
        task_worktrees.activate(graph, kit.worker_id, kit.task['id'])
    assert Path(scope['project']).exists()


@pytest.mark.parametrize('kind', ['worker', 'holder', 'scheduler', 'lease', 'claim', 'area', 'file', 'assignment', 'dirty', 'ignored', 'commit', 'authority'])
@pytest.mark.parametrize('legacy', [False, True])
def test_recovery_refuses_live_or_unique_work(inactive, monkeypatch, kind, legacy):
    kit = inactive
    target = Path(kit.allocation['project'])
    if legacy:
        with GraphStore(kit.base / 'coordination.db') as graph:
            scope = record(graph, 'task-worktree:' + kit.task['id'].split(':')[1], 'task-worktree')
            scope.pop('state')
            graph.upsert_node({'id': scope['id'], 'kind': 'task-worktree', 'attrs': scope})
    if kind in ('worker', 'holder', 'scheduler'):
        pid = {'worker': 555, 'holder': 555, 'scheduler': kit.allocation['scheduler_pid']}[kind]
        monkeypatch.setattr(recovery, 'pid_exists', lambda value: value == pid)
        monkeypatch.setattr(recovery, 'started_at', lambda value: 42)
    elif kind == 'lease':
        with GraphStore(kit.base / 'coordination.db') as graph:
            task = record(graph, kit.task['id'], 'task')
            lease = record(graph, task['lease_id'], 'lease')
            graph.upsert_node({'id': lease['id'], 'kind': 'lease', 'attrs': {**lease, 'active': True}})
    elif kind == 'claim':
        kit.ws.release(kit.parent, 'worktree', str(target))
        kit.ws.claims.claim(kit.ws.auth(kit.child), 'worktree', str(target))
    elif kind == 'file':
        kit.ws.claims.reserve(kit.ws.auth(kit.parent), [('file', str(target / 'code.py'))])
    elif kind in ('area', 'assignment'):
        scope_id = 'task-worktree:' + kit.task['id'].split(':')[1]
        kit.ws.claims.reserve(kit.ws.auth(kit.parent),
                             [('area', str(target / 'code.py'))] if kind == 'area' else [('worktree', str(target))],
                             {'assignment': scope_id if kind == 'area' else 'task-worktree:other',
                              'task': kit.task['id'], 'project': str(target)})
    elif kind in ('dirty', 'ignored'):
        (target / ('code.py' if kind == 'dirty' else '.private-state')).write_text('preserve\n')
    elif kind == 'commit':
        from poolhouse.workspace import integration_git as repo
        (target / 'code.py').write_text('unique = True\n')
        repo.git(target, 'add', '--', 'code.py')
        repo.git(target, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fix: unique task work')
    with pytest.raises((Denied, RuntimeError)):
        recovery.dematerialize(kit.ws, kit.child if kind == 'authority' else kit.parent,
                               kit.task['id'], 'Inactive recovery request')
    assert target.exists()
    assert kit.board.get(kit.parent, kit.task['id'])['state'] == 'blocked'


def test_failed_cleanup_restores_exact_prior_claims(inactive):
    kit = inactive
    target = Path(kit.allocation['project'])
    before = kit.ws.claims._load()
    (target / 'code.py').write_text('pending = True\n')
    with pytest.raises(Denied):
        recovery.dematerialize(kit.ws, kit.parent, kit.task['id'], 'Preserve pending files')
    assert kit.ws.claims._load() == before


def test_registered_legacy_scope_without_state_recovers_through_checked_lifecycle(inactive):
    kit = inactive
    with GraphStore(kit.base / 'coordination.db') as graph:
        scope = record(graph, 'task-worktree:' + kit.task['id'].split(':')[1], 'task-worktree')
        scope.pop('state')
        graph.upsert_node({'id': scope['id'], 'kind': 'task-worktree', 'attrs': scope})
    result = recovery.dematerialize(kit.ws, kit.parent, kit.task['id'], 'Recover verified legacy checkout')
    assert result['cleanup_verified'] and result['state'] == 'blocked'
    with GraphStore(kit.base / 'coordination.db') as graph:
        recovered = record(graph, scope['id'], 'task-worktree')
    assert recovered['state'] == 'reserved' and not Path(scope['project']).exists()


def test_missing_legacy_checkout_preserves_pending_history(inactive):
    from poolhouse import worktreerules
    from poolhouse.workspace import integration_git as repo

    kit = inactive
    with GraphStore(kit.base / 'coordination.db') as graph:
        scope = record(graph, 'task-worktree:' + kit.task['id'].split(':')[1], 'task-worktree')
        scope.pop('state')
        graph.upsert_node({'id': scope['id'], 'kind': 'task-worktree', 'attrs': scope})
    before = kit.board.get(kit.parent, kit.task['id'])
    primary = worktreerules.checkouts(kit.source)[1]
    repo.remove_merged(primary, Path(scope['project']), scope['branch'], repo.git(primary, 'rev-parse', 'HEAD'))
    with pytest.raises(Denied, match='registered checkout'):
        recovery.dematerialize(kit.ws, kit.parent, kit.task['id'], 'Missing legacy checkout')
    after = kit.board.get(kit.parent, kit.task['id'])
    assert after['state'] == before['state'] and after['checkpoints'] == before['checkpoints']


def test_claim_reservation_waits_for_recovery_lock(inactive):
    import threading

    kit = inactive
    with GraphStore(kit.base / 'coordination.db') as graph:
        scope = record(graph, 'task-worktree:' + kit.task['id'].split(':')[1], 'task-worktree')
    entered, finished = threading.Event(), threading.Event()
    def competing():
        entered.set()
        kit.ws.claims.reserve(kit.ws.auth(kit.parent), [('worktree', scope['project'])])
        finished.set()
    with kit.ws.claims.inactive_worktree(kit.ws.auth(kit.parent), scope):
        worker = threading.Thread(target=competing)
        worker.start()
        assert entered.wait(2)
        assert not finished.wait(0.05)
    worker.join(2)
    assert finished.is_set()


def test_existing_legacy_reserved_checkout_keeps_its_baseline(inactive):
    from poolhouse import worktreerules
    from poolhouse.workspace import integration_git as repo

    kit = inactive
    kit.board.resume(kit.parent, kit.task['id'], 'Prepare existing legacy reservation')
    primary = worktreerules.checkouts(kit.source)[1]
    target = Path(kit.allocation['project'])
    moved = primary.parent / 'legacy-task-checkout'
    repo.git(primary, 'worktree', 'move', str(target), str(moved))
    with GraphStore(kit.base / 'coordination.db') as graph:
        scope = record(graph, 'task-worktree:' + kit.task['id'].split(':')[1], 'task-worktree')
        graph.upsert_node({'id': scope['id'], 'kind': 'task-worktree',
                           'attrs': {**scope, 'state': 'reserved', 'project': str(moved)}})
    (primary / 'code.py').write_text('updated = True\n')
    repo.git(primary, 'add', '--', 'code.py')
    repo.git(primary, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fix: advance baseline')
    prepared = task_worktrees.prepare(kit.ws, kit.parent, kit.worker_id, kit.task['id'])
    assert prepared['project'] == str(moved)
    assert prepared['baseline_commit'] == scope['baseline_commit']
