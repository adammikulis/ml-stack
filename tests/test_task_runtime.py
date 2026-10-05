"""Canonical execution renews tokens, preserves leases and stops blocked work."""

import threading
from types import SimpleNamespace

import pytest

from ml_stack.workspace import task_runtime


class Board:
    def __init__(self, project):
        self.ws = SimpleNamespace(base=project)
        self.task = {'state': 'queued', 'limits': {'max_wall_s': 1},
                     'lease': {'resource': {'project': str(project)}}}
        self.calls = []

    def get(self, token, ident):
        return self.task

    def claim(self, token, ident, allocation):
        self.calls.append(('claim', token))
        self.task['state'] = 'working'

    def heartbeat(self, token, ident):
        self.calls.append(('heartbeat', token))

    def checkpoint(self, token, ident, value):
        self.calls.append(('checkpoint', token))

    def submit(self, token, ident, value):
        self.calls.append(('submit', token))
        self.task['state'] = 'review'
        return value

    def block(self, token, ident, reason):
        self.calls.append(('block', token))
        self.task.update(state='blocked', blocked_reason=reason)
        return self.task


def test_reload_token_each_transition_and_use_allocated_project(tmp_path, monkeypatch):
    board, token = Board(tmp_path), ['original']
    monkeypatch.setattr(task_runtime.tokens, 'load', lambda *_: token[0])
    proposal = {'artifacts': {'report.md': 'a' * 64}}

    def run(task, project, stopped, checkpoint):
        assert project == tmp_path.resolve()
        token[0] = 'renewed'
        checkpoint({'summary': 'Native tool completed'})
        return proposal

    assert task_runtime.execute(board, 'worker', 'task', 'allocation', run) == proposal
    assert board.calls == [('claim', 'original'), ('checkpoint', 'renewed'), ('submit', 'renewed')]


def test_heartbeat_failure_cancels_and_blocks_without_resubmission(tmp_path, monkeypatch):
    board = Board(tmp_path)
    monkeypatch.setattr(task_runtime.tokens, 'load', lambda *_: 'token')
    def heartbeat(*_):
        raise RuntimeError('Broker holder expired')
    board.heartbeat = heartbeat

    def run(task, project, stopped, checkpoint):
        for _ in range(50):
            if stopped():
                return {'artifacts': {'report': 'a' * 64}}
            threading.Event().wait(0.01)
        pytest.fail('Native task was not cancelled')

    result = task_runtime.execute(board, 'worker', 'task', 'allocation', run)
    assert result['state'] == 'blocked'
    assert 'Broker holder expired' in result['blocked_reason']
    with pytest.raises(ValueError, match='only a queued'):
        task_runtime.execute(board, 'worker', 'task', 'allocation', run)
    assert board.calls == [('claim', 'token'), ('block', 'token')]


def test_missing_allocated_project_blocks_after_claim(tmp_path, monkeypatch):
    board = Board(tmp_path / 'missing')
    monkeypatch.setattr(task_runtime.tokens, 'load', lambda *_: 'token')
    result = task_runtime.execute(board, 'worker', 'task', 'allocation', lambda *_: pytest.fail('run'))
    assert result['state'] == 'blocked'
