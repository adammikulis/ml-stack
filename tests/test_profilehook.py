"""Automatic harness metadata excludes content and preserves authenticated ownership."""
import io
import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from workspace_kit import Kit, clean_env

from ml_stack import claude, harnessing, profilehook
from ml_stack.workspace import tokens
from ml_stack.workspace.harness_seat import Seat


def payload(event='PreToolUse', **values):
    return {'session_id': 'session-1', 'hook_event_name': event, **values}


def test_hook_metadata_whitelist_does_not_read_content_or_guess_settings():
    result = profilehook.metadata(payload(
        effort={'level': 'xhigh'}, tool_input={'secret': 'private-tool'}, tool_response='private-output',
        transcript_path='/private/transcript', title='private-title', prompt='private-prompt',
        model='model-not-in-this-event', context_window=999999))
    assert result['fields'] == {'harness': 'claude-code', 'effective_effort': 'xhigh'}
    assert 'private' not in json.dumps(result)
    assert 'model-not' not in json.dumps(result)
    assert profilehook.metadata(payload(effort={'level': 'unknown'}))['fields'] == {'harness': 'claude-code'}
    assert profilehook.metadata({'hook_event_name': 'unknown'}) is None


def test_optional_session_model_and_switched_model_are_event_reports_only():
    assert profilehook.metadata(payload('SessionStart'))['fields'] == {'harness': 'claude-code'}
    assert profilehook.metadata(payload('SessionStart', model='qwen'))['fields']['model'] == 'qwen'
    switched = profilehook.metadata(payload('PostModelSwitch', to_model='actual', from_model='old',
                                            requested_model='requested'))
    assert switched['fields']['model'] == 'actual'
    assert 'requested_model' not in switched['fields']


@pytest.mark.parametrize('change', [{'session_id': '/private/transcript'}, {'agent_id': '../foreign'},
                                   {'model': {'nested': 'private'}}, {'model': 'x' * 257}])
def test_malformed_metadata_is_rejected_before_transport(change):
    with pytest.raises(ValueError):
        profilehook.metadata(payload('SessionStart', **change))


def test_helper_sessions_have_separate_stable_namespace_without_actor_selection():
    main = profilehook.metadata(payload('SessionStart', model='main'))
    helper = profilehook.metadata(payload('SessionStart', model='child', agent_id='helper-1', agent_type='Explore'))
    assert main['session'] != helper['session']
    assert helper['session'] == profilehook.metadata(payload(agent_id='helper-1'))['session']
    assert helper['fields']['harness_agent_id'] == 'helper-1'
    assert helper['fields']['harness_agent_type'] == 'Explore'
    assert 'actor' not in helper and 'parent' not in helper


def test_notification_failure_never_blocks_and_redacts_credentials(monkeypatch, capsys):
    secret = 'private-fixture-credential-value'
    monkeypatch.setenv('ML_STACK_WORKSPACE_TOKEN', secret)

    def unavailable(*_args):
        raise RuntimeError(f'daemon unavailable {secret}')

    monkeypatch.setattr(profilehook, 'send', unavailable)
    assert profilehook.run(['observe', '--label', 'alice', '--root', '/tmp'],
                           io.StringIO(json.dumps(payload(effort={'level': 'high'})))) == 0
    output = capsys.readouterr()
    assert output.out == ''
    assert 'daemon unavailable' in output.err
    assert secret not in output.err


@pytest.mark.parametrize('body', ['{', 'x' * (profilehook.MAX_INPUT + 1)], ids=('malformed', 'oversized'))
def test_notification_parser_is_bounded_and_fails_open(body, monkeypatch, capsys):
    monkeypatch.setattr(profilehook, 'send', lambda *_args: pytest.fail('invalid metadata transported'))
    assert profilehook.run(['observe', '--label', 'alice', '--root', '/tmp'], io.StringIO(body)) == 0
    assert capsys.readouterr().out == ''


def test_metadata_authentication_preserves_actor_and_separates_helper_observations(monkeypatch, tmp_path):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    token = kit.agent('alice')
    tokens.store(kit.base, 'alice', token)
    monkeypatch.setattr(profilehook.harness_remote, 'context', lambda *_args, **_kwargs: None)
    profilehook.notify(payload('SessionStart', model='main'), 'alice', tmp_path)
    profilehook.notify(payload('SessionStart', model='helper', agent_id='helper-1'), 'alice', tmp_path)
    rows = kit.ws.execution_profiles(token)
    assert len(rows) == 2 and {row['actor'] for row in rows} == {'alice'}
    assert len({row['session'] for row in rows}) == 2
    assert {row['fields']['model']['value'] for row in rows} == {'main', 'helper'}
    assert all(row['fields']['model']['source'] == 'reported' for row in rows)
    assert all(row['parent'] is None for row in rows)


def test_launcher_observation_preserves_requests_without_inventing_effective_limits(monkeypatch, tmp_path):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    token = kit.agent('alice')
    tokens.store(kit.base, 'alice', token)
    monkeypatch.setattr(profilehook.harness_remote, 'context', lambda *_args, **_kwargs: None)
    args = SimpleNamespace(model='requested', ctx=32000, effort='high', max_output_tokens=1000, max_turns=None)
    seat = Seat('alice', persistent=True, base=kit.base)
    assert seat.record_execution(args, 'claude-code', ('http://localhost:9', 'served', 16000), tmp_path)
    fields = kit.ws.execution_profiles(token)[0]['fields']
    assert fields['requested_context']['value'] == 32000
    assert fields['requested_output_tokens']['value'] == 1000
    assert fields['effective_context'] == {'value': None, 'source': 'unknown'}
    assert fields['effective_effort'] == {'value': None, 'source': 'unknown'}
    assert fields['effective_turns'] == {'value': None, 'source': 'unknown'}
    assert fields['model']['value'] == 'served'
    assert args.max_turns is None and args.ctx == 32000


def test_claude_metadata_hooks_use_supported_command_schema_and_keep_auth_hooks():
    configured = json.loads(claude.settings('authorize', 'notify', 300, 'cleanup', 'observe'))
    hooks = configured['hooks']
    assert hooks['PreToolUse'][0]['hooks'][0]['command'] == 'authorize'
    assert hooks['PreToolUse'][0]['hooks'][0]['timeout'] == 330
    assert hooks['Stop'][0]['hooks'][0]['command'] == 'cleanup'
    for event in profilehook.EVENTS:
        metadata = hooks[event][-1]['hooks'][0]
        assert metadata == {'type': 'command', 'command': 'observe', 'timeout': 2, 'async': True}
    command = harnessing.hook_command('observe', role='read-only', label='alice', root=Path('/tmp'), protect=[])
    assert 'ml_stack.profilehook' in command and '--label alice' in command


@pytest.mark.slow
def test_standalone_metadata_hook_command_records_actual_fixture_session(monkeypatch, tmp_path):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    token = kit.agent('alice')
    tokens.store(kit.base, 'alice', token)
    root = tmp_path / 'project'
    root.mkdir()
    command = harnessing.hook_command('observe', role='read-only', label='alice', root=root, protect=[])
    environment = {**os.environ, 'ML_STACK_WORKSPACE_HOME': str(kit.base),
                   'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    completed = subprocess.run(shlex.split(command), input=json.dumps(payload(effort={'level': 'high'})),
                               env=environment, capture_output=True, text=True, timeout=10, check=False)
    assert completed.returncode == 0 and completed.stdout == ''
    assert completed.stderr == ''
    fields = kit.ws.execution_profiles(token)[0]['fields']
    assert fields['effective_effort'] == {'value': 'high', 'source': 'reported'}


def test_model_switch_clears_old_effort_until_new_effective_observation(monkeypatch, tmp_path):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    token = kit.agent('alice')
    tokens.store(kit.base, 'alice', token)
    monkeypatch.setattr(profilehook.harness_remote, 'context', lambda *_args, **_kwargs: None)
    profilehook.notify(payload('SessionStart', model='old', effort={'level': 'high'}), 'alice', tmp_path)
    profilehook.notify(payload('PostModelSwitch', to_model='new'), 'alice', tmp_path)
    latest = kit.ws.execution_profiles(token)[-1]['fields']
    assert latest['model']['value'] == 'new'
    assert latest['effective_effort'] == {'value': None, 'source': 'unknown'}
    assert latest['model_version'] == {'value': None, 'source': 'unknown'}
    profilehook.notify(payload(effort={'level': 'low'}), 'alice', tmp_path)
    assert kit.ws.execution_profiles(token)[-1]['fields']['effective_effort'] == {'value': 'low', 'source': 'reported'}


def test_launcher_metadata_outage_has_an_independent_bounded_deadline(monkeypatch, tmp_path):
    import threading
    import time

    from ml_stack.workspace import harness_seat

    release = threading.Event()
    finished = threading.Event()
    monkeypatch.setattr(harness_seat, 'PROFILE_TIMEOUT', 0.05)

    def unavailable(*_args):
        try:
            release.wait(5)
            return False
        finally:
            finished.set()

    monkeypatch.setattr(Seat, '_record_execution', unavailable)
    args = SimpleNamespace(ctx=32000, max_turns=None)
    seat = Seat('alice', persistent=True)
    start = time.monotonic()
    try:
        assert not seat.record_execution(args, 'claude-code', ('http://localhost:9', 'served', 16000), tmp_path)
        assert time.monotonic() - start < 0.5
        assert args.ctx == 32000 and args.max_turns is None
    finally:
        release.set()
        assert finished.wait(2)


def test_observer_runtime_metadata_requires_matching_installed_module_and_full_commit(monkeypatch, tmp_path):
    marker = tmp_path / 'built-from'
    marker.write_text('a' * 40)
    module = Path(profilehook.__file__).resolve()
    monkeypatch.setattr(profilehook.sys, 'prefix', str(module.parents[2]))
    installed = SimpleNamespace(version='0.1.0', locate_file=lambda name: module if name.endswith('profilehook.py') else marker)
    monkeypatch.setattr(profilehook, 'distribution', lambda _name: installed)
    facts = profilehook.runtime_facts()
    assert facts['runtime_commit'] == 'a' * 40 and facts['runtime_version'] == '0.1.0'
    marker.write_text('invalid')
    assert 'runtime_commit' not in profilehook.runtime_facts()
    installed.locate_file = lambda _name: tmp_path / 'editable-source.py'
    assert profilehook.runtime_facts() == {'python_version': profilehook.sys.version.split()[0]}


def test_local_launcher_observation_keeps_its_established_workspace_authority(monkeypatch, tmp_path):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    token = kit.agent('alice')
    tokens.store(kit.base, 'alice', token)
    monkeypatch.setattr(profilehook, 'send', lambda *_args: pytest.fail('local seat authority was reselected'))
    seat = Seat('alice', persistent=True, base=kit.base)
    assert seat.record_execution(SimpleNamespace(ctx=32000), 'claude-code',
                                 ('http://localhost:9', 'served', 16000), tmp_path)
    assert kit.ws.execution_profiles(token)[0]['actor'] == 'alice'
