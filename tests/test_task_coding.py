"""Canonical native coding uses bounded turns and cancels its owned process."""

import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ml_stack.workspace import localagent, task_coding


@pytest.mark.redteam
def test_native_turn_and_output_limits_preserve_authority(monkeypatch):
    agent = localagent.Agent('worker', 'qwen', profile='coding', harness='claude',
                             effort='high', max_effort='low')
    seen = {}
    def process(self, turn, command, environment, context):
        seen.update(command=command, environment=environment)
        return 0
    monkeypatch.setattr(task_coding.Manager, '_process', process)
    manager = task_coding.TaskManager(None, SimpleNamespace(base=None), agent)
    context = (None, 'claude', 'prompt', None)
    manager._process(None, ['claude', '--print'], {'AUTHORITY': 'unchanged'}, context)
    assert seen['command'][-2:] == ['--max-turns', '60']
    assert 'CLAUDE_CODE_MAX_OUTPUT_TOKENS' not in seen['environment']
    assert seen['environment']['CLAUDE_CODE_EFFORT_LEVEL'] == 'low'
    assert seen['environment']['AUTHORITY'] == 'unchanged'
    with pytest.raises(ValueError, match='bounded Claude'):
        manager._process(None, ['codex'], {}, (None, 'codex', 'prompt', None))


@pytest.mark.parametrize('effort, thinking', [('off', False), ('medium', True)])
def test_native_template_thinking_is_separate_from_output_and_stream(monkeypatch, effort, thinking):
    agent = localagent.Agent('worker', 'Qwen3.8-27B.gguf', harness='claude', effort=effort)
    seen = {}
    monkeypatch.setattr(task_coding.Manager, '_process',
                        lambda self, turn, command, environment, context: seen.update(environment) or 0)
    manager = task_coding.TaskManager(None, SimpleNamespace(base=None), agent)
    manager._process(None, ['claude'], {'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '12000',
        'CLAUDE_CODE_EXTRA_BODY': json.dumps({'stream': True, 'max_tokens': 12000,
            'chat_template_kwargs': {'custom': 'kept', 'enable_thinking': not thinking}})},
        (None, 'claude', 'prompt', None))
    body = json.loads(seen['CLAUDE_CODE_EXTRA_BODY'])
    assert body == {'stream': True, 'max_tokens': 12000,
                    'chat_template_kwargs': {'custom': 'kept', 'enable_thinking': thinking}}
    assert seen['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] == '12000'
    assert 'MAX_THINKING_TOKENS' not in seen


def test_native_effort_switch_has_no_generated_output_or_thinking_cap(monkeypatch):
    agent = localagent.Agent('worker', '/cache/Qwen3.8-27B-UD-Q4_K_XL.gguf', harness='claude', effort='off')
    seen = {}
    monkeypatch.setattr(task_coding.Manager, '_process',
                        lambda self, turn, command, environment, context: seen.update(environment) or 0)
    manager = task_coding.TaskManager(None, SimpleNamespace(base=None), agent)
    context = (None, 'claude', '', None)
    manager._process(None, ['claude'], {}, context)
    assert json.loads(seen['CLAUDE_CODE_EXTRA_BODY'])['chat_template_kwargs']['enable_thinking'] is False
    assert 'MAX_THINKING_TOKENS' not in seen and 'CLAUDE_CODE_MAX_OUTPUT_TOKENS' not in seen
    manager.agent = replace(agent, effort='medium')
    manager._process(None, ['claude'], seen.copy(), context)
    assert json.loads(seen['CLAUDE_CODE_EXTRA_BODY'])['chat_template_kwargs']['enable_thinking'] is True
    manager._process(None, ['claude'], {'MAX_THINKING_TOKENS': '0'}, context)
    assert seen['MAX_THINKING_TOKENS'] == '0'
    assert json.loads(seen['CLAUDE_CODE_EXTRA_BODY'])['chat_template_kwargs']['enable_thinking'] is False


@pytest.mark.parametrize('body', ['broken', '[]', '{"chat_template_kwargs":[]}'])
def test_invalid_native_extra_body_does_not_launch(monkeypatch, body):
    monkeypatch.setattr(task_coding.Manager, '_process', lambda *_: pytest.fail('native launched'))
    manager = task_coding.TaskManager(None, SimpleNamespace(base=None), localagent.Agent('worker', 'qwen'))
    with pytest.raises(ValueError):
        manager._process(None, ['claude'], {'CLAUDE_CODE_EXTRA_BODY': body}, (None, 'claude', '', None))


def test_canonical_stop_cancels_native_turn(tmp_path, monkeypatch):
    stop = threading.Event()
    cancelled = []
    conversation = SimpleNamespace(id='conversation')
    monkeypatch.setattr(task_coding, 'Conversations', lambda *_: SimpleNamespace(start=lambda **_: conversation))
    monkeypatch.setattr(task_coding.git, 'head', lambda _: 'a' * 40)
    monkeypatch.setattr(task_coding.la, 'folder', lambda _: tmp_path)
    monkeypatch.setattr(task_coding.tokens, 'load', lambda *_: 'token')
    monkeypatch.setattr(task_coding.work_reputation, 'brief', lambda *_: 'Recorded reputation data')
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


def test_proposal_binds_committed_files_and_deletions(tmp_path):
    git = task_coding.git
    git.run(['init', str(tmp_path)])
    git.run(['config', 'user.name', 'Test'], cwd=tmp_path)
    git.run(['config', 'user.email', 'test@example.invalid'], cwd=tmp_path)
    (tmp_path / 'deleted.py').write_text('OLD = True\n')
    git.run(['add', 'deleted.py'], cwd=tmp_path)
    git.run(['commit', '-m', 'baseline'], cwd=tmp_path)
    baseline = git.head(tmp_path)
    (tmp_path / 'deleted.py').unlink()
    (tmp_path / 'new.py').write_text('NEW = True\n')
    git.run(['add', '--', 'deleted.py', 'new.py'], cwd=tmp_path)
    git.run(['commit', '-m', 'task changes'], cwd=tmp_path)
    task = {'lease': {'resource': {'baseline_commit': baseline}}}
    turn = SimpleNamespace(text='Changed code and verified checks.', session='session')
    proposal = task_coding._proposal(localagent.Agent('worker', 'qwen'), task, tmp_path, turn, 'python')
    assert set(proposal['artifacts']) == {'.task.patch', '.task-report.md', 'new.py'}
    assert 'deleted file mode' in (tmp_path / '.task.patch').read_text()
    assert proposal['provenance']['commit'] == git.head(tmp_path)
    assert not git.run(['status', '--porcelain'], cwd=tmp_path).stdout.strip()
    (tmp_path / 'new.py').write_text('DIRTY = True\n')
    with pytest.raises(RuntimeError, match='uncommitted changes'):
        task_coding._proposal(localagent.Agent('worker', 'qwen'), task, tmp_path, turn, 'python')
