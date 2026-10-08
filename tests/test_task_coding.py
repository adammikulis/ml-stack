"""Native coding uses bounded turns and cancels its owned process."""

import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ml_stack.workspace import localagent, task_coding


@pytest.fixture
def native(tmp_path):
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps({'hooks': {'PreToolUse': [{'matcher': '*', 'hooks': [
        {'type': 'command', 'command': 'python -m ml_stack.harnesshook pre --role read-only'}]}]}}))
    return ['claude', '--settings', str(settings)], (tmp_path, 'claude', 'prompt', None)


@pytest.mark.redteam
def test_native_turn_and_output_limits_preserve_authority(monkeypatch, native):
    agent = localagent.Agent('worker', 'qwen', profile='coding', harness='claude',
                             effort='high', max_effort='low')
    seen = {}
    def process(self, turn, command, environment, context):
        seen.update(command=command, environment=environment)
        return 0
    monkeypatch.setattr(task_coding.Manager, '_process', process)
    manager = task_coding.TaskManager(None, SimpleNamespace(base=None), agent)
    command, context = native
    manager._process(None, [*command, '--print'], {'AUTHORITY': 'unchanged'}, context)
    assert '--max-turns' not in seen['command']
    settings = json.loads(context[0].joinpath('task-settings.json').read_text())
    hooks = settings['hooks']['PreToolUse']
    assert '--protect' in hooks[0]['hooks'][0]['command']
    assert 'ml_stack.workspace.task_caps' in hooks[1]['hooks'][0]['command']
    assert seen['command'][seen['command'].index('--system-prompt') + 1] == task_coding.BOOTSTRAP
    assert seen['command'][seen['command'].index('--tools') + 1] == 'Read,Edit,Write,Bash,Glob,Grep,Agent'
    assert '--dangerously-skip-permissions' not in seen['command']
    assert 'CLAUDE_CODE_MAX_OUTPUT_TOKENS' not in seen['environment']
    assert seen['environment']['CLAUDE_CODE_EFFORT_LEVEL'] == 'low'
    assert seen['environment']['AUTHORITY'] == 'unchanged'
    manager._process(None, ['pi', '--mode', 'json'], {'AUTHORITY': 'unchanged'},
                     (None, 'pi', 'prompt', None))
    assert seen['command'] == ['pi', '--mode', 'json']
    assert seen['environment']['AUTHORITY'] == 'unchanged'


@pytest.mark.parametrize('effort, thinking', [('off', False), ('medium', True)])
def test_native_template_thinking_is_separate_from_output_and_stream(monkeypatch, native, effort, thinking):
    agent = localagent.Agent('worker', 'Qwen3.8-27B.gguf', harness='claude', effort=effort)
    seen = {}
    monkeypatch.setattr(task_coding.Manager, '_process',
                        lambda self, turn, command, environment, context: seen.update(environment) or 0)
    manager = task_coding.TaskManager(None, SimpleNamespace(base=None), agent)
    command, context = native
    manager._process(None, command, {'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '12000',
        'CLAUDE_CODE_EXTRA_BODY': json.dumps({'stream': True, 'max_tokens': 12000,
            'chat_template_kwargs': {'custom': 'kept', 'enable_thinking': not thinking}})},
        context)
    body = json.loads(seen['CLAUDE_CODE_EXTRA_BODY'])
    assert body == {'stream': True, 'max_tokens': 12000,
                    'chat_template_kwargs': {'custom': 'kept', 'enable_thinking': thinking}}
    assert seen['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] == '12000'
    assert 'MAX_THINKING_TOKENS' not in seen


def test_native_effort_switch_has_no_generated_output_or_thinking_cap(monkeypatch, native):
    agent = localagent.Agent('worker', '/cache/Qwen3.8-27B-UD-Q4_K_XL.gguf', harness='claude', effort='off')
    seen = {}
    monkeypatch.setattr(task_coding.Manager, '_process',
                        lambda self, turn, command, environment, context: seen.update(environment) or 0)
    manager = task_coding.TaskManager(None, SimpleNamespace(base=None), agent)
    command, context = native
    manager._process(None, command, {}, context)
    assert json.loads(seen['CLAUDE_CODE_EXTRA_BODY'])['chat_template_kwargs']['enable_thinking'] is False
    assert 'MAX_THINKING_TOKENS' not in seen and 'CLAUDE_CODE_MAX_OUTPUT_TOKENS' not in seen
    manager.agent = replace(agent, effort='medium')
    manager._process(None, command, seen.copy(), context)
    assert json.loads(seen['CLAUDE_CODE_EXTRA_BODY'])['chat_template_kwargs']['enable_thinking'] is True
    manager._process(None, command, {'MAX_THINKING_TOKENS': '0'}, context)
    assert seen['MAX_THINKING_TOKENS'] == '0'
    assert json.loads(seen['CLAUDE_CODE_EXTRA_BODY'])['chat_template_kwargs']['enable_thinking'] is False


@pytest.mark.parametrize('body', ['broken', '[]', '{"chat_template_kwargs":[]}'])
def test_invalid_native_extra_body_does_not_launch(monkeypatch, native, body):
    monkeypatch.setattr(task_coding.Manager, '_process', lambda *_: pytest.fail('native launched'))
    manager = task_coding.TaskManager(None, SimpleNamespace(base=None), localagent.Agent('worker', 'qwen'))
    command, context = native
    with pytest.raises(ValueError):
        manager._process(None, command, {'CLAUDE_CODE_EXTRA_BODY': body}, context)


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
        assert 'Recorded reputation data' not in prompt
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
    git.run(['config', 'core.autocrlf', 'true'], cwd=tmp_path)
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
    task_coding.repo.reviewed_files(tmp_path, git.head(tmp_path), proposal['artifacts'])
    assert not git.run(['status', '--porcelain'], cwd=tmp_path).stdout.strip()
    (tmp_path / 'new.py').write_text('DIRTY = True\n')
    with pytest.raises(RuntimeError, match='uncommitted changes'):
        task_coding._proposal(localagent.Agent('worker', 'qwen'), task, tmp_path, turn, 'python')


def test_canonical_pi_launcher_receives_independent_budget_and_clamped_effort(tmp_path, monkeypatch):
    from ml_stack import coding
    from ml_stack.fleet.conversations import Conversations

    project = tmp_path / 'project'
    project.mkdir()
    store = Conversations(tmp_path / 'chats')
    conversation = store.start(model='qwen', settings={
        'mode': 'coding', 'harness': 'pi', 'project': str(project),
        'effort': 'high', 'max_effort': 'low', 'max_output_tokens': 32000})
    seen = {}
    def launch(model, role, project, **options):
        seen.update(options)
        return 0
    monkeypatch.setattr(coding, 'launch_coding_agent', launch)
    agent = localagent.Agent('worker', 'qwen', profile='coding', harness='pi', max_output_tokens=32000)
    manager = task_coding.TaskManager(store, SimpleNamespace(base=None), agent)
    manager._run(task_coding.Turn(conversation.id), conversation, 'Implement a queue')
    assert seen['max_output_tokens'] == 32000
    assert seen['effort'] == 'low'
    assert seen['max_turns'] is None


def test_unlimited_native_tool_counter_preserves_parser_bounds(tmp_path):
    from ml_stack.workspace import task_caps
    path = tmp_path / "counter.json"
    path.write_text(json.dumps({'version': 1, 'calls': 10000, 'limit': None}))
    assert task_caps.admit(path)
    assert json.loads(path.read_text())["calls"] == 10001
    path.write_text(json.dumps({'version': 1, 'calls': 0, 'limit': 2}))
    assert task_caps.admit(path) and task_caps.admit(path)
    assert not task_caps.admit(path)
    path.write_text(json.dumps({'version': 1, 'calls': True, 'limit': None}))
    with pytest.raises(ValueError):
        task_caps.admit(path)
