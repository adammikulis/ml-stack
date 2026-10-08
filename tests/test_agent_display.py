"""Real registry presentation never grants authority or erases parentage."""
import pytest
from workspace_kit import Kit, clean_env

from ml_stack.workspace.agent_display import metadata
from ml_stack.workspace.identity import Denied

PERSON = {'terminal': (True, True), 'env': {}}

pytest_plugins = ["test_workspace_remote"]


@pytest.fixture
def kit(monkeypatch, tmp_path):
    return Kit(clean_env(monkeypatch, tmp_path))


def test_unknown_sessions_are_distinct_and_ordinals_survive_revoke_and_device_label_changes(kit):
    registry = kit.ws.registry
    first = kit.agent('codex-first')
    second = kit.agent('codex-second')
    one = metadata(registry, 'codex-first')
    two = metadata(registry, 'codex-second')
    assert one['display_name'] != two['display_name']
    assert not one['coordinator_eligible']
    registry.register_session(first)
    registry.register_session(second)
    kit.ws.set_model('codex-second', 'claude-sonnet-5-5', **PERSON)
    saved = registry.info('codex-second')['presentation'].copy()
    kit.ws.revoke(kit.owner, 'codex-first')
    kit.agent('codex-third')
    metadata(registry, 'codex-third')
    registry.record_device_claim(second, {'os': 'Windows', 'hostname': 'changed'})
    assert registry.info('codex-second')['presentation'] == saved
    assert metadata(registry, 'codex-second')['coordinator_eligible']
    assert metadata(registry, 'owner')['display_name'] == 'owner'


def test_authenticated_child_cannot_promote_and_labels_do_not_grant_main_status(kit):
    token = kit.agent('codex-main')
    registry = kit.ws.registry
    kit.ws.register_session(token)
    child = registry.delegate(kit.ws.auth(token), 'task', 600, ('read',), 10)
    with pytest.raises(Denied, match='top-level'):
        registry.register_session(child)
    shown = metadata(registry, 'codex-main/task')
    assert shown['display_name'].startswith('Subagent · task (parent Model unknown')
    assert not shown['coordinator_eligible']
    activity = metadata(registry, 'codex-main', 'integration')
    assert activity['session_kind'] == 'main'
    assert 'activity integration' in activity['display_name']
    assert not activity['coordinator_eligible']
    kit.ws.claim_model(token, 'gpt-6', label='helper')
    helper = metadata(registry, 'codex-main', 'helper')
    assert helper['session_kind'] == 'helper'
    assert helper['display_name'].startswith('Subagent · helper (parent Model unknown')


def test_revoked_main_is_ineligible(kit):
    token = kit.agent('codex-main')
    kit.ws.register_session(token)
    kit.ws.revoke(kit.owner, 'codex-main')
    assert not metadata(kit.ws.registry, 'codex-main')['coordinator_eligible']
    with pytest.raises(Denied):
        kit.ws.registry.register_session(token)


def test_presented_agents_children_and_internal_senders_are_read_only(kit, monkeypatch):
    from ml_stack.workspace.service import GREETER

    token = kit.agent('main-worker')
    kit.ws.register_session(token)
    kit.ws.set_model('main-worker', 'claude-sonnet-5-5', **PERSON)
    kit.ws.registry.delegate(kit.ws.auth(token), 'helper', 600, ('read',), 10)

    def refuse_presentation(name):
        raise AssertionError(f'unexpected presentation mutation for {name}')

    monkeypatch.setattr(kit.ws.registry, 'ensure_presentation', refuse_presentation)
    assert metadata(kit.ws.registry, GREETER.id)['session_kind'] == 'unknown'
    assert metadata(kit.ws.registry, 'owner')['session_kind'] == 'person'
    assert metadata(kit.ws.registry, 'main-worker')['coordinator_eligible']
    child = metadata(kit.ws.registry, 'main-worker/helper')
    assert child['session_kind'] == 'subagent'
    assert not child['coordinator_eligible']


def test_canonical_registration_preserves_model_and_rights_and_refuses_child(host):
    from test_workspace_remote import call, joined

    from ml_stack.workspace.coordinator_calls import READS, WRITES

    first = joined(host, name='codex-first')
    before = call(host, first, 'whoami')[1]['result']
    code, shown = call(host, first, 'register_session', device={'os': 'macOS', 'hostname': 'test'})
    assert code == 200
    shown = shown['result']
    assert shown['display_name'] == 'Model unknown · Mac · session 1'
    assert not shown['coordinator_eligible']
    assert shown['coordinator_reason'] == 'model not in the tier table'
    after = call(host, first, 'whoami')[1]['result']
    assert (after['model'], after['can'], after['parent']) == (before['model'], before['can'], before['parent'])
    assert call(host, first, 'register_session')[1]['result'] == shown
    child = call(host, first, 'delegate', 'review')[1]['result']
    assert call(host, child, 'register_session')[0] == 403
    assert 'main-session' in WRITES and 'main-session' not in READS


def test_canonical_brief_uses_authenticated_parent_instead_of_codex_alias(kit, monkeypatch, capsys):
    from types import SimpleNamespace

    from ml_stack.workspace import cli

    token = kit.agent('codex-session')
    monkeypatch.setattr(cli, '_context', lambda args: (kit.ws, token))
    cli._brief(SimpleNamespace(agent='codex', name='review', registered=False), None)
    output = capsys.readouterr().out
    assert '--agent codex-session --label review' in output
    assert '--agent codex --label' not in output
    assert 'do not elect yourself coordinator' in output


def test_unknown_ordinals_do_not_collide_after_device_metadata_arrives(kit):
    tokens = [kit.agent(name) for name in ('codex-one', 'codex-two')]
    registry = kit.ws.registry
    metadata(registry, 'codex-one')
    registry.record_device_claim(tokens[0], {'os': 'macOS', 'hostname': 'same'})
    registry.record_device_claim(tokens[1], {'os': 'macOS', 'hostname': 'same'})
    assert metadata(registry, 'codex-one')['display_name'] != metadata(registry, 'codex-two')['display_name']


def test_expired_and_restricted_main_have_no_coordinator_eligibility(kit):
    token = kit.agent('codex-expiring', ttl_s=10)
    kit.ws.register_session(token)
    now = kit.ws.clock()
    kit.ws.registry.clock = lambda: now + 11
    assert not metadata(kit.ws.registry, 'codex-expiring')['coordinator_eligible']
    with pytest.raises(Denied):
        kit.ws.registry.register_session(token)
    kit.ws.registry.clock = kit.ws.clock
    token = kit.agent('codex-restricted')
    agents = kit.ws.registry._load()
    agents['codex-restricted']['can'] = ['read']
    kit.ws.registry._save(agents)
    with pytest.raises(Denied, match='claim'):
        kit.ws.register_session(token)
    assert not metadata(kit.ws.registry, 'codex-restricted')['coordinator_eligible']


def test_cli_main_session_uses_canonical_mutation_rpc(host, monkeypatch, capsys):
    from types import SimpleNamespace

    from test_workspace_remote import call, joined

    from ml_stack.workspace import cli, project_connection

    agent = joined(host, name='codex-cli')

    def invoke(operation, token, *args, **kwargs):
        assert token == agent['token']
        code, result = call(host, agent, operation, *args, **kwargs)
        assert code == 200, result
        return result['result']

    workspace = project_connection.CanonicalWorkspace(SimpleNamespace(call=invoke), agent['token'])
    monkeypatch.setattr(cli, '_project_connection', lambda: {'host': 'canonical'})
    monkeypatch.setattr(cli, '_context', lambda args, connection: (workspace, agent['token']))
    handler = next(entry[3] for entry in cli.TABLE if entry[0] == 'main-session')
    args = SimpleNamespace(cmd='main-session', json=True, request_id='', harness='codex')
    assert cli._runner(handler)(args) == 0
    assert '"session_kind": "main"' in capsys.readouterr().out


@pytest.mark.parametrize("model,harness,expected", [
    ("gpt-6", "codex", "ChatGPT"),
    ("claude-sonnet-4-6", "claude-code", "Claude"),
    ("Qwen3.8-Flash-Next-GSQ-RCO-Coder", "codex", "Qwen"),
    ("thinkingcap-qwen3.8-27b", "codex", "Qwen"),
    ("unknown-provider-model", "claude-code", "Model unknown"),
])
def test_readable_family_uses_recorded_model_not_harness_or_auth_id(kit, model, harness, expected):
    token = kit.agent('codex-opaque-session')
    kit.ws.claim_model(token, model, harness)
    shown = metadata(kit.ws.registry, 'codex-opaque-session')
    assert shown['display_name'].startswith(expected + ' · ')
    assert 'codex-opaque-session' not in shown['display_name']
    assert not shown['coordinator_eligible']
    before = kit.ws.registry.info('codex-opaque-session')
    kit.ws.register_session(token)
    assert kit.ws.registry.info('codex-opaque-session')['can'] == before['can']
    assert kit.ws.registry.info('codex-opaque-session')['model'] == model


def test_same_family_native_sessions_get_distinct_stable_ordinals(kit):
    registry = kit.ws.registry
    first = kit.agent('claude-code-' + 'a' * 32)
    second = kit.agent('claude-code-' + 'b' * 32)
    for token in (first, second):
        kit.ws.claim_model(token, 'claude-sonnet-4-6', 'claude-code')
        kit.ws.register_session(token)
    names = ['claude-code-' + digit * 32 for digit in ('a', 'b')]
    before = [registry.info(name)['presentation'].copy() for name in names]
    assert before[0]['ordinal'] != before[1]['ordinal']
    assert metadata(registry, names[0])['display_name'] != metadata(registry, names[1])['display_name']
    kit.ws.claim_model(second, 'Qwen3.8-Flash', 'codex')
    assert [registry.info(name)['presentation'] for name in names] == before
    assert metadata(registry, names[1])['display_name'].startswith('Qwen · ')


def test_registration_records_unknown_model_harness_without_rewriting_model_history(host):
    from test_workspace_remote import call, joined

    agent = joined(host, name='native-unknown-model')
    before = call(host, agent, 'whoami')[1]['result']
    code, result = call(host, agent, 'register_session', harness='claude-code')
    assert code == 200, result
    after = call(host, agent, 'whoami')[1]['result']
    assert (after['model'], after['model_state'], after['models']) == (before['model'], before['model_state'], before['models'])
    assert (after['can'], after['parent'], after['project']) == (before['can'], before['parent'], before['project'])
    assert after['harness'] == 'claude-code' and after['harness_state'] == 'claimed'
    child = call(host, agent, 'delegate', 'helper')[1]['result']
    assert call(host, child, 'register_session', harness='codex')[0] == 403


def test_harness_registration_preserves_launcher_verified_model_and_harness(kit):
    token = kit.agent('native-verified')
    registry = kit.ws.registry
    registry.record_model('native-verified', 'Qwen3.8-Flash', 'codex', 'verified')
    before = registry.info('native-verified')
    with pytest.raises(Denied, match='launcher'):
        kit.ws.register_session(token, harness='claude-code')
    kit.ws.register_session(token, harness='codex')
    after = registry.info('native-verified')
    assert (after['model'], after['models'], after['harness']) == (before['model'], before['models'], before['harness'])
    assert after['harness_state'] == 'verified'


def test_verified_model_does_not_promote_reported_harness_on_repeat_registration(kit):
    token = kit.agent('model-only-verified')
    registry = kit.ws.registry
    registry.record_model('model-only-verified', 'Qwen3.8-Flash', '', 'verified')
    before = registry.info('model-only-verified')
    for _ in range(2):
        kit.ws.register_session(token, harness='codex')
        info = registry.info('model-only-verified')
        assert info['harness_state'] == 'claimed'
        assert kit.ws.whoami_model('model-only-verified')['harness_state'] == 'claimed'
        assert (info['model'], info['model_state'], info['models']) == (before['model'], before['model_state'], before['models'])
    kit.ws.register_session(token, harness='claude-code')
    assert registry.info('model-only-verified')['harness_state'] == 'claimed'
    assert registry.info('model-only-verified')['harness'] == 'claude-code'


def test_launcher_harness_observation_updates_independent_confidence(kit):
    token = kit.agent('observed-harness')
    registry = kit.ws.registry
    kit.ws.register_session(token, harness='claude-code')
    assert registry.info('observed-harness')['harness_state'] == 'claimed'
    registry.record_model('observed-harness', 'Qwen3.8-Flash', 'codex', 'verified')
    assert registry.info('observed-harness')['harness_state'] == 'verified'
    with pytest.raises(Denied, match='launcher'):
        kit.ws.register_session(token, harness='claude-code')
