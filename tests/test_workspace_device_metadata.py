"""Device provenance follows actual actors without adding permission or account authority."""
import pytest
from workspace_kit import Kit, clean_env

from ml_stack.workspace import device_metadata, onboard
from ml_stack.workspace.identity import Denied


@pytest.fixture
def kit(monkeypatch, tmp_path):
    return Kit(clean_env(monkeypatch, tmp_path))


def device(system='macOS'):
    return {'device_id': 'a' * 64, 'hostname': 'workstation', 'os': system,
            'verification': 'local-observed', 'source': 'local-runtime'}


def test_local_device_observation_is_cached_and_provider_failure_stays_unknown(monkeypatch):
    calls = []
    monkeypatch.setattr(device_metadata, 'device_id', lambda: calls.append(True) or 'a' * 64)
    monkeypatch.setattr(device_metadata.platform, 'system', lambda: 'Darwin')
    device_metadata.current.cache_clear()
    try:
        assert device_metadata.current()['os'] == 'macOS'
        assert device_metadata.current()['device_id'] == 'a' * 64 and calls == [True]
        device_metadata.current.cache_clear()
        def unavailable():
            raise RuntimeError('no native machine identity')
        monkeypatch.setattr(device_metadata, 'device_id', unavailable)
        assert device_metadata.current()['device_id'] is None
        assert device_metadata.current()['verification'] == 'unknown'
    finally:
        device_metadata.current.cache_clear()


def test_device_metadata_is_visible_and_delegates_inherit_without_changing_rights(kit):
    token = kit.agent('lead', 'lead')
    kit.ws.registry._record_device('lead', device())
    before = kit.ws.auth(token)
    child = kit.ws.delegate(token, 'helper')
    info = kit.ws.registry.info(child['id'])
    assert info['device']['label'] == 'macOS · workstation'
    assert kit.ws.auth(token) == before
    row = next(row for row in kit.ws.registered() if row['id'] == child['id'])
    assert row['device'] == info['device']
    assert 'hash' not in row and 'token' not in row


@pytest.mark.redteam
def test_agent_device_report_cannot_claim_paired_authority_or_change_another_actor(kit):
    alice, bob = kit.agent('alice'), kit.agent('bob')
    kit.ws.registry._record_device('bob', device('Windows'))
    before = kit.ws.registry.info('bob')
    kit.ws.registry.record_device_claim(alice, {**device(), 'verification': 'paired',
                                               'peer_id': 'b' * 64, 'source': 'fleet-pairing'})
    own = kit.ws.registry.info('alice')['device']
    assert own['verification'] == 'agent-reported' and own['peer_id'] is None
    assert kit.ws.registry.info('bob') == before and kit.ws.auth(bob).id == 'bob'
    with pytest.raises(Denied):
        kit.ws.registry.record_device_claim('', device())


@pytest.mark.parametrize('value', [None, [], {'device_id': 'invented'},
                                  {**device(), 'hostname': 'bad\nname'},
                                  {**device(), 'verification': 'paired'}])
def test_malformed_or_unproved_device_metadata_is_rejected(kit, value):
    kit.agent('alice')
    before = kit.ws.registry.info('alice')['device']
    with pytest.raises(ValueError):
        kit.ws.registry._record_device('alice', value)
    assert kit.ws.registry.info('alice')['device'] == before


def test_subagent_brief_names_actual_runtime_device_without_prompt_history(monkeypatch):
    monkeypatch.setattr(device_metadata, 'current', lambda: device_metadata.normalize(device()))
    text = onboard.brief('helper', 'lead')
    assert 'macOS · workstation (local-observed)' in text
    assert 'provenance grants no permissions' in text
