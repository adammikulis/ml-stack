"""Authenticated parent polling claims accepted reviews once before publication."""

import threading

import pytest
from taskboard_kit import accepted, board, proposed

from poolhouse.graph.store import GraphStore
from poolhouse.workspace import task_scheduler
from poolhouse.workspace.identity import Denied

__all__ = ['board']


def complete(kit):
    proposed(kit)
    kit.board.review(kit.parent, kit.task['id'], accepted())


def attempts(kit):
    with GraphStore(kit.ws.base / 'coordination.db') as graph:
        return [row['attrs'] for row in graph.nodes('integration-attempt')]


def test_parent_polls_accepted_review_once_and_records_graph_links(board, monkeypatch):
    complete(board)
    calls = []
    monkeypatch.setattr(task_scheduler.task_integration, 'integrate',
                        lambda ws, token, task: calls.append((token, task)) or {'state': 'published', 'commit': 'a' * 40})
    assert task_scheduler.integrate_completed(board.ws, board.parent, board.worker_id)[0]['state'] == 'published'
    assert task_scheduler.integrate_completed(board.ws, board.parent, board.worker_id) == []
    assert calls == [(board.parent, board.task['id'])]
    record = attempts(board)[0]
    assert record['owner'] == 'lead' and record['state'] == 'published'
    assert record['review_hash'] == board.board.get(board.parent, board.task['id'])['review']['review_hash']


@pytest.mark.parametrize('raises', [False, True])
def test_blocked_publication_is_not_retried(board, monkeypatch, raises):
    complete(board)
    calls = []
    def publish(*args):
        calls.append(args)
        if raises:
            raise RuntimeError('Development ownership unavailable')
        return {'state': 'blocked', 'reason': 'Required check failed'}
    monkeypatch.setattr(task_scheduler.task_integration, 'integrate', publish)
    assert task_scheduler.integrate_completed(board.ws, board.parent, board.worker_id)[0]['state'] == 'blocked'
    assert task_scheduler.integrate_completed(board.ws, board.parent, board.worker_id) == []
    assert len(calls) == 1 and attempts(board)[0]['state'] == 'blocked'


def test_proposed_work_is_not_published_before_independent_review(board, monkeypatch):
    proposed(board)
    monkeypatch.setattr(task_scheduler.task_integration, 'integrate', lambda *_: pytest.fail('unreviewed publication'))
    assert task_scheduler.integrate_completed(board.ws, board.parent, board.worker_id) == []
    assert attempts(board) == []
    with pytest.raises(Denied, match='registered worker parent'):
        task_scheduler.integrate_completed(board.ws, board.child, board.worker_id)


def test_concurrent_pollers_reserve_review_before_running_helper(board, monkeypatch):
    complete(board)
    started, release = threading.Event(), threading.Event()
    calls = []
    def publish(*args):
        calls.append(args)
        started.set()
        assert release.wait(10)
        return {'state': 'published'}
    monkeypatch.setattr(task_scheduler.task_integration, 'integrate', publish)
    results = []
    thread = threading.Thread(target=lambda: results.append(task_scheduler.integrate_completed(
        board.ws, board.parent, board.worker_id)))
    thread.start()
    try:
        assert started.wait(10)
        assert attempts(board)[0]['state'] == 'started'
        assert task_scheduler.integrate_completed(board.ws, board.parent, board.worker_id) == []
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and len(calls) == 1
    assert results == [[{'state': 'published'}]]


def test_dead_integration_owner_becomes_visible_recovery_without_blind_retry(board, monkeypatch):
    complete(board)
    review = board.board.get(board.parent, board.task['id'])['review']
    from poolhouse.workspace.task_graph import save
    with GraphStore(board.base / 'coordination.db') as graph:
        save(graph, 'integration-attempt', {'id': 'integration-attempt:' + review['review_hash'],
            'task': board.task['id'], 'worker': board.worker_id, 'owner': 'lead',
            'review_hash': review['review_hash'], 'state': 'started', 'started': 1000,
            'owner_pid': 999999, 'owner_started': 42})
    monkeypatch.setattr(task_scheduler, 'pid_exists', lambda pid: False)
    monkeypatch.setattr(task_scheduler.task_integration, 'integrate', lambda *_: pytest.fail('unsafe retry'))
    result = task_scheduler.integrate_completed(board.ws, board.parent, board.worker_id)
    assert result[0]['state'] == 'blocked'
    assert attempts(board)[0]['recovery_required']
    assert board.board.get(board.parent, board.task['id'])['state'] == 'accepted'
    assert task_scheduler.integrate_completed(board.ws, board.parent, board.worker_id) == []


def test_chat_assignment_review_finishes_without_git_scope(board):
    from dataclasses import replace

    from poolhouse.workspace import localagent
    complete(board)
    runner = localagent.load(board.ws, 'native-worker')
    localagent.save(board.ws, replace(runner, profile='chat'))
    spec = dict(board.spec, title='Answer a question', source_key='chat:question', capabilities=['chat'])
    task = board.board.create(board.parent, spec)
    allocation = task_scheduler.assign_next(board.ws, board.parent, board.worker_id, 'native-grant')
    assert allocation['task'] == task['id']
    assert allocation['worker'] == board.worker_id
    board.board.claim(board.child, task['id'], allocation['allocation_id'])
    board.board.submit(board.child, task['id'], {
        'artifacts': {'answer.txt': 'a' * 64},
        'checks': [{'name': 'Answer reviewed', 'passed': True}],
        'summary': 'Question answered',
        'provenance': {'environment': 'chat', 'model': 'qwen', 'runtime': 'Agents SDK'}})
    board.board.review(board.parent, task['id'], accepted())
    assert board.board.get(board.parent, task['id'])['state'] == 'completed'
    with GraphStore(board.base / 'coordination.db') as graph:
        assert not any(row['attrs']['task'] == task['id'] for row in graph.nodes('task-worktree'))
