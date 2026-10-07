"""Real registry presentation never grants authority or erases parentage."""
import pytest
from workspace_kit import Kit, clean_env

from ml_stack.workspace.agent_display import metadata
from ml_stack.workspace.identity import Denied

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
    assert shown['display_name'].startswith('Subagent · task (parent Codex')
    assert not shown['coordinator_eligible']
    activity = metadata(registry, 'codex-main', 'integration')
    assert activity['session_kind'] == 'main'
    assert 'activity integration' in activity['display_name']
    assert not activity['coordinator_eligible']
    kit.ws.claim_model(token, 'gpt-6', label='helper')
    helper = metadata(registry, 'codex-main', 'helper')
    assert helper['session_kind'] == 'helper'
    assert helper['display_name'].startswith('Subagent · helper (parent Codex')


def test_revoked_main_is_ineligible(kit):
    token = kit.agent('codex-main')
    kit.ws.register_session(token)
    kit.ws.revoke(kit.owner, 'codex-main')
    assert not metadata(kit.ws.registry, 'codex-main')['coordinator_eligible']
    with pytest.raises(Denied):
        kit.ws.registry.register_session(token)


def test_canonical_registration_preserves_model_and_rights_and_refuses_child(host):
    from test_workspace_remote import call, joined

    from ml_stack.workspace.coordinator_calls import READS, WRITES

    first = joined(host, name='codex-first')
    before = call(host, first, 'whoami')[1]['result']
    code, shown = call(host, first, 'register_session', device={'os': 'macOS', 'hostname': 'test'})
    assert code == 200
    shown = shown['result']
    assert shown['display_name'] == 'Codex · Mac · session 1'
    assert shown['coordinator_eligible']
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
    cli._brief(SimpleNamespace(agent='codex', name='review'), None)
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
    args = SimpleNamespace(cmd='main-session', json=True, request_id='')
    assert cli._runner(handler)(args) == 0
    assert '"session_kind": "main"' in capsys.readouterr().out
