"""Canonical native coding uses bounded turns and cancels its owned process."""

import threading
from types import SimpleNamespace

import pytest

from ml_stack.workspace import localagent, task_coding


def test_native_turn_and_output_limits_preserve_authority(monkeypatch):
    agent = localagent.Agent('worker', 'qwen', profile='coding', harness='claude',
                             effort='high', max_effort='low')
    seen = {}
    def process(self, turn, command, environment, context):
        seen.update(command=command, environment=environment)
        return 0
    monkeypatch.setattr(task_coding.BoundManager, '_process', process)
    manager = task_coding.TaskManager(None, SimpleNamespace(base=None), agent)
    context = (None, 'claude', 'prompt', None)
    manager._process(None, ['claude', '--print'], {'AUTHORITY': 'unchanged'}, context)
    assert seen['command'][-2:] == ['--max-turns', '60']
    assert seen['environment']['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] == '4096'
    assert seen['environment']['CLAUDE_CODE_EFFORT_LEVEL'] == 'low'
    assert seen['environment']['AUTHORITY'] == 'unchanged'
    with pytest.raises(ValueError, match='bounded Claude'):
        manager._process(None, ['codex'], {}, (None, 'codex', 'prompt', None))


def test_canonical_stop_cancels_native_turn(tmp_path, monkeypatch):
    stop = threading.Event()
    cancelled = []
    conversation = SimpleNamespace(id='conversation')
    monkeypatch.setattr(task_coding, 'Conversations', lambda *_: SimpleNamespace(start=lambda **_: conversation))
    monkeypatch.setattr(task_coding.git, 'head', lambda _: 'a' * 40)
    monkeypatch.setattr(task_coding.la, 'folder', lambda _: tmp_path)
    monkeypatch.setattr(task_coding.Turn, 'cancel', lambda self: (cancelled.append(True), self.cancelled.set()))
    def run(self, turn, conversation, prompt):
        assert 'calling done' not in prompt
        assert 'Linux testing is on hold' in prompt
        stop.set()
        assert turn.cancelled.wait(1)
    monkeypatch.setattr(task_coding.TaskManager, '_run', run)
    task = {'id': 'task:' + 'a' * 32, 'title': 'Fix queue', 'description': 'Fix the queue', 'acceptance': ['Fails correctly']}
    with pytest.raises(RuntimeError, match='cancelled'):
        task_coding.perform(SimpleNamespace(base=tmp_path), localagent.Agent('worker', 'qwen'),
                            task, tmp_path, (stop.is_set, lambda _: None))
    assert cancelled == [True]
