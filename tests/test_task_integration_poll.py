"""Authenticated parent polling claims accepted reviews once before publication."""

import threading

import pytest
from taskboard_kit import accepted, board, proposed

from ml_stack.graph.store import GraphStore
from ml_stack.workspace import task_scheduler
from ml_stack.workspace.identity import Denied

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
