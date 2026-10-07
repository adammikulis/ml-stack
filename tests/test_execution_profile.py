"""Authenticated bounded observations preserve independent metadata provenance."""
import pytest
from workspace_kit import Kit, clean_env

from ml_stack.workspace import execution_profile, tokens
from ml_stack.workspace.identity import Denied


@pytest.fixture
def kit(monkeypatch, tmp_path):
    return Kit(clean_env(monkeypatch, tmp_path))


def document(event='event-1', **fields):
    return {'session': 'session-1', 'event_id': event, 'event': 'SessionStart', 'fields': fields}


def test_own_profile_fields_are_reported_and_unobserved_limits_unknown(kit):
    alice = kit.agent('alice')
    before = kit.ws.registry.info('alice')
    row = kit.ws.record_execution_profile(alice, document(model='qwen', effective_effort='high',
                                                        requested_context=32000, requested_output_tokens=900))
    assert row['actor'] == 'alice'
    assert row['fields']['model'] == {'value': 'qwen', 'source': 'reported'}
    assert row['fields']['effective_effort'] == {'value': 'high', 'source': 'reported'}
    assert row['fields']['effective_context'] == {'value': None, 'source': 'unknown'}
    assert row['fields']['requested_output_tokens']['value'] == 900
    assert row['fields']['requested_context']['value'] == 32000
    assert kit.ws.registry.info('alice') == before
    assert kit.ws.execution_profiles(alice) == [row]
    assert kit.ws.execution_profiles(kit.agent('bob')) == []


@pytest.mark.parametrize('extra', [{'actor': 'bob'}, {'parent': 'owner'}, {'source': 'verified'},
                                  {'task': 'task1'}, {'device': 'owned'}, {'tool_input': {}}])
def test_observation_payload_cannot_select_identity_authority_or_content(kit, extra):
    alice = kit.agent('alice')
    with pytest.raises(ValueError):
        kit.ws.record_execution_profile(alice, {**document(model='qwen'), **extra})
    assert kit.ws.execution_profiles(alice) == []


@pytest.mark.parametrize('fields', [{'model': {'value': 'x', 'source': 'verified'}}, {'effort': 'high'},
                                   {'effective_context': True}, {'effective_wall_seconds': float('inf')},
                                   {'model': 'x\nsecret'}, {'runtime_commit': 'invented'},
                                   {'requested_output_tokens': -1}, {'prompt': 'private'}])
def test_profile_metadata_whitelist_rejects_malformed_values(kit, fields):
    with pytest.raises(ValueError):
        kit.ws.record_execution_profile(kit.agent('alice'), document(**fields))


def test_profile_events_are_idempotent_bounded_and_cannot_be_rebound(kit, monkeypatch):
    monkeypatch.setattr(execution_profile, 'HISTORY', 3)
    alice = kit.agent('alice')
    first = kit.ws.record_execution_profile(alice, document(model='a'))
    assert kit.ws.record_execution_profile(alice, document(model='a')) == first
    with pytest.raises(ValueError, match='different metadata'):
        kit.ws.record_execution_profile(alice, document(model='b'))
    duplicate = kit.ws.record_execution_profile(alice, document('duplicate-state', model='a'))
    assert duplicate['event_id'] == 'duplicate-state'
    with pytest.raises(ValueError, match='different metadata'):
        kit.ws.record_execution_profile(alice, document('duplicate-state', model='b'))
    for number in range(2, 6):
        kit.ws.record_execution_profile(alice, document(f'event-{number}', model=str(number)))
    rows = kit.ws.execution_profiles(alice)
    assert len(rows) == 3
    assert rows[-1]['fields']['model']['value'] == '5'


def test_child_observations_preserve_registry_parent_without_impersonation(kit):
    alice = kit.agent('alice')
    delegated = kit.ws.delegate(alice, 'child')
    child = tokens.load(kit.ws.base, delegated['id'])
    row = kit.ws.record_execution_profile(child, document(model='qwen'))
    assert row['parent'] == 'alice'
    assert row['actor'] != 'alice'
    assert kit.ws.execution_profiles(alice) == []


def test_revoked_agent_and_redirected_graph_are_refused(kit, tmp_path):
    alice = kit.agent('alice')
    target = tmp_path / 'outside'
    target.mkdir()
    (kit.ws.base / 'execution-profiles').symlink_to(target, target_is_directory=True)
    with pytest.raises(Denied):
        kit.ws.record_execution_profile(alice, document(model='qwen'))
    assert list(target.iterdir()) == []
    with pytest.raises(Denied):
        kit.ws.record_execution_profile('', document(model='qwen'))
    (kit.ws.base / 'execution-profiles').unlink()
    kit.ws.revoke(kit.owner, 'alice')
    with pytest.raises(Denied):
        kit.ws.record_execution_profile(alice, document(model='qwen'))


def test_restricted_child_cannot_record_without_existing_send_right(kit):
    alice = kit.agent('alice')
    child = kit.ws.delegate(alice, 'reader', can=('read',))
    token = tokens.load(kit.ws.base, child['id'])
    with pytest.raises(Denied, match='right to send'):
        kit.ws.record_execution_profile(token, document(model='qwen'))
    assert kit.ws.execution_profiles(token) == []


def test_expired_child_cannot_record_observations(kit, monkeypatch):
    alice = kit.agent('alice')
    child = kit.ws.delegate(alice, 'short', ttl_s=1)
    token = tokens.load(kit.ws.base, child['id'])
    monkeypatch.setattr(kit.ws.registry, 'clock', lambda: child['expires'] + 1)
    with pytest.raises(Denied):
        kit.ws.record_execution_profile(token, document(model='qwen'))
    assert not (kit.ws.base / 'execution-profiles').exists()
