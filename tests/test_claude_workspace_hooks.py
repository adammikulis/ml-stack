"""Claude event hooks record claimed models and provide nonblocking subagent context."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ml_stack.net import git
from ml_stack.workspace import Workspace, guide, project

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / 'scripts/hooks'


@pytest.fixture
def commands(tmp_path, monkeypatch):
    directory = tmp_path / 'bin'
    directory.mkdir()
    executable = directory / 'ml-stack-workspace'
    executable.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
path = Path(os.environ['HOOK_CALLS'])
calls = json.loads(path.read_text()) if path.exists() else []
calls.append(dict(argv=sys.argv[1:], session=os.environ.get('ML_STACK_SESSION_ID'), harness=os.environ.get('ML_STACK_SESSION_HARNESS')))
path.write_text(json.dumps(calls))
if os.environ.get('HOOK_FAILURE'):
    print(os.environ['HOOK_FAILURE'], file=sys.stderr)
    raise SystemExit(3)
print('Authenticated parent brief: hello-model; claims and independent review required')
''')
    executable.chmod(0o700)
    binary = shutil.which('git')
    assert binary, 'hook tests require installed Git'
    (directory / 'git').symlink_to(binary)
    calls = tmp_path / 'calls.json'
    monkeypatch.setenv('PATH', str(directory))
    monkeypatch.setenv('HOOK_CALLS', str(calls))
    monkeypatch.delenv('ANTHROPIC_MODEL', raising=False)
    return directory, calls


def invoke(name, event):
    return subprocess.run([sys.executable, str(HOOKS / name)], input=json.dumps(event),
                          text=True, capture_output=True, timeout=40, check=False)


@pytest.mark.parametrize('model', ['claude-sonnet-4-6', {'id': 'claude-opus-4-6'}, 'Qwen/Qwen3-Coder-30B'])
def test_session_event_forwards_exact_model_as_a_claim(commands, model):
    _, calls = commands
    done = invoke('claude-session-start', {'model': model, 'session_id': 'native-main-1'})
    assert done.returncode == 0 and not done.stdout and not done.stderr
    records = json.loads(calls.read_text())
    assert records[0]['session'] == 'native-main-1' and records[0]['harness'] == 'claude-code'
    assert records[1]['argv'] == ['main-session', '--agent', 'claude-code', '--harness', 'claude-code']
    assert records[0]['argv'] == ['whoami', '--agent', 'claude-code', '--model',
                                           model['id'] if isinstance(model, dict) else model,
                                           '--harness', 'claude-code']


def test_session_uses_reported_environment_model_when_event_omits_it(commands, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_MODEL', 'claude-haiku-4-5')
    _, calls = commands
    assert invoke('claude-session-start', {}).returncode == 0
    assert json.loads(calls.read_text())[0]['argv'][4] == 'claude-haiku-4-5'


def test_missing_model_registers_session_without_guessing_and_explains_it(commands):
    _, calls = commands
    done = invoke('claude-session-start', {'session_id': 'native-with-unknown-model'})
    assert done.returncode == 0 and not done.stdout
    records = json.loads(calls.read_text())
    assert records[0] == {'argv': ['whoami', '--agent', 'claude-code', '--harness', 'claude-code'],
                          'session': 'native-with-unknown-model', 'harness': 'claude-code'}
    assert records[1]['argv'] == ['main-session', '--agent', 'claude-code', '--harness', 'claude-code']
    assert 'model unavailable' in done.stderr and 'no model claim recorded' in done.stderr


@pytest.mark.parametrize('name', ['claude-session-start', 'claude-subagent-start'])
@pytest.mark.redteam
def test_unavailable_workspace_is_nonblocking_with_redacted_reason(commands, monkeypatch, tmp_path, name):
    monkeypatch.setenv('HOOK_FAILURE', 'Denied: daemon unavailable mlws1.actor.syntheticSecret123 mlsk1.syntheticCluster123')
    done = invoke(name, {'model': 'claude-sonnet-4-6', 'cwd': str(tmp_path)})
    assert done.returncode == 0 and 'command exited 3' in done.stderr and 'daemon unavailable' in done.stderr
    assert 'syntheticSecret123' not in done.stderr and 'syntheticCluster123' not in done.stderr
    assert '[REDACTED:workspace-token]' in done.stderr and '[REDACTED:cluster-token]' in done.stderr


@pytest.mark.parametrize('name', ['claude-session-start', 'claude-subagent-start'])
def test_missing_workspace_executable_is_nonblocking_and_explained(commands, name, tmp_path):
    directory, _ = commands
    (directory / 'ml-stack-workspace').unlink()
    done = invoke(name, {'model': 'claude-sonnet-4-6', 'cwd': str(tmp_path)})
    assert done.returncode == 0 and 'ml-stack-workspace' in done.stderr and 'No such file' in done.stderr


def test_subagent_event_includes_brief_and_actual_primary_branch_rules(commands, tmp_path):
    repository = tmp_path / 'project'
    git.run(['init', '-b', 'development', str(repository)])
    done = invoke('claude-subagent-start', {'agent_type': 'Explore', 'agent_id': 'abcdef12345', 'session_id': 'native-parent', 'cwd': str(repository)})
    assert done.returncode == 0 and not done.stderr
    output = json.loads(done.stdout)['hookSpecificOutput']
    assert output['hookEventName'] == 'SubagentStart'
    context = output['additionalContext']
    assert 'Authenticated parent brief: hello-model' in context
    assert f'primary checkout {repository.resolve()} is on the development branch development' in context
    assert 'Acquire authenticated claims before mutation' in context and 'separate sibling worktrees' in context
    records = json.loads(commands[1].read_text())
    assert len(records) == 1
    assert records[0] == {'argv': ['brief', 'explore-abcdef', '--agent', 'claude-code'],
                          'session': 'native-parent', 'harness': 'claude-code'}


@pytest.mark.parametrize('stage', ['SessionStart', 'SubagentStart'])
def test_timeout_warning_is_nonblocking_and_redacts_the_command(monkeypatch, capsys, stage):
    spec = importlib.util.spec_from_file_location('tested_workspace_hook', HOOKS / 'workspace_hook.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(['ml-stack-workspace', 'mlws1.actor.syntheticSecret123'], 30)
    monkeypatch.setattr(module.subprocess, 'run', timeout)
    assert module.run(['ml-stack-workspace', 'whoami'], stage) is None
    warning = capsys.readouterr().err
    assert stage in warning and 'timed out' in warning and 'syntheticSecret123' not in warning


@pytest.mark.parametrize('name', ['claude-session-start', 'claude-subagent-start'])
def test_malformed_events_warn_and_do_not_run_commands(commands, name):
    done = subprocess.run([sys.executable, str(HOOKS / name)], input='not JSON', text=True,
                          capture_output=True, timeout=40, check=False)
    assert done.returncode == 0 and not done.stdout and 'Expecting value' in done.stderr
    assert not commands[1].exists()


def test_repository_claude_settings_wire_the_maintained_event_hooks():
    settings = json.loads((ROOT / '.claude/settings.json').read_text())
    for event, name in [('SessionStart', 'claude-session-start'), ('SubagentStart', 'claude-subagent-start')]:
        hooks = settings['hooks'][event][0]['hooks']
        assert hooks == [{'type': 'command', 'command': f'$CLAUDE_PROJECT_DIR/scripts/hooks/{name}'}]


def test_actual_hooks_record_claimed_metadata_and_authenticated_subagent_brief(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', str(Path(sys.executable).parent), prepend=os.pathsep)
    monkeypatch.delenv('CODEX_THREAD_ID', raising=False)
    monkeypatch.delenv('CODEX_SESSION_ID', raising=False)
    monkeypatch.delenv('ML_STACK_AGENT', raising=False)
    monkeypatch.setenv('ML_STACK_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('ML_STACK_WORKSPACE_HOME', str(tmp_path / 'ws'))
    monkeypatch.setenv('PYTHONPATH', str(ROOT / 'src'))
    repository = tmp_path / 'project'
    git.run(['init', '-b', 'development', str(repository)])
    git.run(['-c', 'user.name=Hook fixture', '-c', 'user.email=fixture@example.invalid',
             'commit', '--allow-empty', '-m', 'fixture project'], cwd=repository)
    ws = Workspace(tmp_path / 'ws')
    guide.agent_connect(ws, 'claude-code', project.describe(str(repository)), host_coordinator=False)
    monkeypatch.chdir(repository)
    done = invoke('claude-session-start', {'model': 'claude-sonnet-4-6'})
    assert done.returncode == 0 and not done.stderr
    info = ws.registry.info('claude-code')
    assert info['model'] == 'claude-sonnet-4-6' and info['model_state'] == 'claimed'
    assert info['harness'] == 'claude-code'
    brief = invoke('claude-subagent-start', {'agent_type': 'Explore', 'agent_id': 'abcdef12345', 'cwd': str(repository)})
    assert brief.returncode == 0 and not brief.stderr
    context = json.loads(brief.stdout)['hookSpecificOutput']['additionalContext']
    assert 'hello-model' in context and '--agent claude-code' in context and 'explore-abcdef' in context
    assert 'Acquire authenticated claims before mutation' in context


def test_session_exports_preserve_native_context_without_global_configuration(commands, tmp_path, monkeypatch):
    root = tmp_path / '.claude'
    directory = root / 'session-env' / 'native-main-1'
    directory.mkdir(parents=True)
    target = directory / 'exports.sh'
    target.write_text('export EXISTING_NATIVE_CONTEXT=preserved\n')
    target.chmod(0o600)
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(root))
    monkeypatch.setenv('CLAUDE_ENV_FILE', str(target))
    done = invoke('claude-session-start', {'session_id': 'native-main-1'})
    assert done.returncode == 0 and 'model unavailable' in done.stderr
    text = target.read_text()
    assert 'EXISTING_NATIVE_CONTEXT=preserved' in text
    assert 'export ML_STACK_SESSION_ID=native-main-1' in text
    assert 'export ML_STACK_SESSION_HARNESS=claude-code' in text
    assert [item['argv'][0] for item in json.loads(commands[1].read_text())] == ['whoami', 'main-session']
    shell = subprocess.run(['/bin/sh', '-c', '. "$1"; ml-stack-workspace whoami --agent claude-code',
                            'hook-test', str(target)], capture_output=True, text=True, check=False)
    assert shell.returncode == 0
    recorded = json.loads(commands[1].read_text())[-1]
    assert recorded['session'] == 'native-main-1' and recorded['harness'] == 'claude-code'


@pytest.mark.redteam
@pytest.mark.parametrize('unsafe', ['foreign-session', 'symlink'])
def test_session_exports_refuse_unbound_or_symlinked_files(commands, tmp_path, monkeypatch, unsafe):
    root = tmp_path / '.claude'
    directory = root / 'session-env' / 'native-main-1'
    directory.mkdir(parents=True)
    target = directory / 'exports.sh'
    protected = tmp_path / 'protected'
    protected.write_text('untouched')
    if unsafe == 'symlink':
        target.symlink_to(protected)
    else:
        target = protected
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(root))
    monkeypatch.setenv('CLAUDE_ENV_FILE', str(target))
    done = invoke('claude-session-start', {'session_id': 'native-main-1', 'model': 'claude-sonnet-4-6'})
    assert done.returncode == 0 and 'workspace SessionStart:' in done.stderr
    assert protected.read_text() == 'untouched'


def test_child_event_never_registers_main_session(commands):
    done = invoke('claude-session-start', {'session_id': 'parent', 'agent_id': 'child',
                                         'model': 'claude-sonnet-4-6'})
    assert done.returncode == 0
    assert [item['argv'][0] for item in json.loads(commands[1].read_text())] == ['whoami']


@pytest.mark.parametrize('session', ['has space', 'line\nfeed', 'unicode-é', '', 'x' * 257],
                         ids=['whitespace', 'control', 'nonascii', 'empty', 'oversized'])
def test_invalid_native_session_does_not_export_or_register(commands, tmp_path, monkeypatch, session):
    target = tmp_path / 'exports.sh'
    monkeypatch.setenv('CLAUDE_ENV_FILE', str(target))
    done = invoke('claude-session-start', {'session_id': session, 'model': 'claude-sonnet-4-6'})
    assert done.returncode == 0 and 'invalid native session_id' in done.stderr
    assert not target.exists() and not commands[1].exists()
