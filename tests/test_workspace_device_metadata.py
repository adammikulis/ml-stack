"""Device provenance follows actual actors without adding permission or account authority."""
import pytest
from workspace_kit import Kit, clean_env

from poolhouse.workspace import device_metadata, onboard, tokens
from poolhouse.workspace.identity import Denied


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
                                               'peer_id': 'b' * 64, 'peer_verification': 'paired', 'source': 'fleet-pairing'})
    own = kit.ws.registry.info('alice')['device']
    assert own['verification'] == 'agent-reported' and own['peer_id'] is None and own['peer_verification'] == 'unknown'
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


def test_child_missing_profile_derives_parent_facts_without_registry_writes(kit):
    token = kit.agent('parent')
    kit.ws.registry._record_device('parent', device())
    kit.ws.claim_model(token, 'Qwen3.8-Flash', 'codex')
    child = kit.ws.delegate(token, 'helper')['id']
    registry = kit.ws.registry
    stored = registry._load()
    stored[child]['device'] = {}
    registry._save(stored)
    before = registry.path.read_bytes()
    shown = registry.info(child)
    assert shown['device']['hostname'] == 'workstation'
    assert shown['device']['verification'] == 'inherited'
    assert shown['device']['parent_verification'] == 'local-observed'
    assert shown['device']['source'] == 'parent-registry'
    assert shown['device']['inherited_from'] == 'parent'
    assert shown['harness'] == 'codex' and shown['harness_state'] == 'inherited'
    assert registry.path.read_bytes() == before
    assert shown['parent'] == 'parent'
    assert shown['can'] == stored[child]['can']


def test_child_own_profile_wins_and_caller_cannot_forge_inheritance(kit):
    token = kit.agent('parent')
    kit.ws.registry._record_device('parent', device())
    child = kit.ws.delegate(token, 'helper')
    registry = kit.ws.registry
    child_token = tokens.load(kit.ws.base, child['id'])
    assert child_token
    registry.record_device_claim(child_token, {**device('Windows'),
                                 'inherited_from': 'unrelated', 'source': 'parent-registry',
                                 'verification': 'inherited', 'parent_verification': 'paired'})
    shown = registry.info(child['id'])
    assert shown['device']['os'] == 'Windows'
    assert shown['device']['verification'] == 'agent-reported'
    assert shown['device']['source'] == 'agent-report'
    assert shown['device']['inherited_from'] == ''
    assert shown['device']['parent_verification'] == 'unknown'
    assert registry.info('parent')['device']['os'] == 'macOS'
    registry.record_device_claim(child_token, device())
    own = registry.info(child['id'])['device']
    assert own['os'] == 'macOS' and own['hostname'] == 'workstation'
    assert own['verification'] == 'agent-reported' and own['inherited_from'] == ''
    kit.ws.claim_model(child_token, 'claude-sonnet-4-6', 'claude-code')
    own_info = registry.info(child['id'])
    assert own_info['harness'] == 'claude-code' and own_info['harness_state'] == 'claimed'


def test_unknown_profile_is_nonempty_and_missing_parent_does_not_invent_device(kit):
    kit.agent('worker')
    registry = kit.ws.registry
    stored = registry._load()
    stored['worker']['device'] = {}
    stored['worker']['parent'] = 'missing-parent'
    registry._save(stored)
    before = registry.path.read_bytes()
    shown = registry.info('worker')
    assert shown['device'] and shown['device']['label'] == 'Unknown device'
    assert shown['device']['verification'] == 'unknown'
    assert shown['device']['hostname'] == ''
    assert shown['harness'] == ''
    assert registry.path.read_bytes() == before


def test_own_profile_updates_child_harness_without_promoting_or_changing_model(kit):
    token = kit.agent('parent')
    kit.ws.registry._record_device('parent', device())
    child = kit.ws.delegate(token, 'helper')
    own = tokens.load(kit.ws.base, child['id'])
    before = kit.ws.registry.info(child['id'])
    shown = kit.ws.record_profile(own, None, 'codex')
    assert shown['id'] == child['id']
    assert shown['device']['inherited_from'] == 'parent'
    assert shown['harness'] == 'codex' and shown['harness_state'] == 'claimed'
    for key in ('parent', 'can', 'project', 'expires', 'model', 'model_state', 'presentation'):
        assert shown[key] == before[key]
    with pytest.raises(Denied, match='top-level'):
        kit.ws.register_session(own)
    again = kit.ws.record_profile(own, None, '')
    assert again['harness'] == 'codex'


@pytest.mark.redteam
def test_own_profile_report_is_atomic_when_verified_harness_refuses_change(kit):
    token = kit.agent('worker')
    kit.ws._record_model('worker', 'Qwen3.8-Flash', 'codex', 'verified')
    before = kit.ws.registry.info('worker')
    with pytest.raises(Denied, match='launcher'):
        kit.ws.record_profile(token, device('Windows'), 'claude-code')
    assert kit.ws.registry.info('worker') == before
    shown = kit.ws.record_profile(token, device(), 'codex')
    assert shown['harness_state'] == 'verified'
    assert shown['device']['verification'] == 'agent-reported'
    assert shown['model'] == 'Qwen3.8-Flash'


@pytest.mark.redteam
@pytest.mark.parametrize('report', ['claim', 'profile'])
def test_an_agent_report_never_sets_the_stable_device_or_machine_identity(kit, report):
    token = kit.agent('worker')
    forged = {**device(), 'machine_id': 'b' * 16, 'peer_id': 'c' * 64, 'peer_verification': 'paired',
              'verification': 'paired', 'source': 'fleet-pairing'}
    if report == 'claim':
        kit.ws.registry.record_device_claim(token, forged)
    else:
        kit.ws.record_profile(token, forged, '')
    shown = kit.ws.registry.info('worker')['device']
    assert shown['device_id'] is None and shown['machine_id'] is None and shown['peer_id'] is None
    assert shown['verification'] == 'agent-reported' and shown['peer_verification'] == 'unknown'
    assert shown['hostname'] == 'workstation'


@pytest.mark.redteam
@pytest.mark.parametrize('stronger', [
    {'verification': 'local-observed', 'source': 'local-runtime'},
    {'verification': 'agent-reported', 'source': 'agent-report', 'peer_id': 'd' * 64, 'peer_verification': 'paired'},
    {'verification': 'paired', 'source': 'fleet-pairing', 'peer_id': 'd' * 64, 'peer_verification': 'paired'},
])
@pytest.mark.parametrize('report', ['claim', 'profile'])
def test_a_profile_report_does_not_replace_a_stronger_record(kit, stronger, report):
    token = kit.agent('worker')
    kit.ws.registry._record_device('worker', {**device(), **stronger})
    before = kit.ws.registry.info('worker')['device']
    if report == 'claim':
        kit.ws.registry.record_device_claim(token, device('Windows'))
    else:
        kit.ws.record_profile(token, device('Windows'), 'codex')
    assert kit.ws.registry.info('worker')['device'] == before


def test_a_report_may_replace_an_inherited_or_an_earlier_agent_report(kit):
    token = kit.agent('parent')
    kit.ws.registry._record_device('parent', device())
    child = kit.ws.delegate(token, 'helper')
    own = tokens.load(kit.ws.base, child['id'])
    assert kit.ws.registry.info(child['id'])['device']['verification'] == 'inherited'
    kit.ws.record_profile(own, device('macOS'), '')
    kit.ws.record_profile(own, device('Windows'), '')
    shown = kit.ws.registry.info(child['id'])['device']
    assert shown['os'] == 'Windows' and shown['verification'] == 'agent-reported'


def test_runtime_commit_stays_and_the_installed_revision_duplicate_is_gone():
    shown = device_metadata.normalize({**device(), 'runtime_commit': 'a' * 40, 'machine_id': 'b' * 16})
    assert shown['runtime_commit'] == 'a' * 40 and shown['machine_id'] == 'b' * 16
    assert 'installed_revision' not in shown
    assert {'inherited_from', 'parent_verification', 'peer_verification'} <= set(shown)


@pytest.mark.parametrize('field,value', [('machine_id', 'invalid'), ('runtime_commit', 'abc'),
                                         ('observed_at', float('nan')), ('observed_at', True),
                                         ('architecture', 'bad\nvalue'), ('peer_verification', 'paired')])
def test_extended_profile_fields_are_bounded(field, value):
    with pytest.raises(ValueError):
        device_metadata.normalize({**device(), field: value})


def test_autofill_names_wsl_and_fleet_machine_identity(monkeypatch):
    monkeypatch.setattr(device_metadata, 'device_id', lambda: 'a' * 64)
    monkeypatch.setattr(device_metadata, 'machine_id', lambda: 'b' * 16)
    monkeypatch.setattr(device_metadata.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(device_metadata.platform, 'release', lambda: '6.6-microsoft-standard-WSL2')
    monkeypatch.setattr(device_metadata.platform, 'machine', lambda: 'x86_64')
    device_metadata.current.cache_clear()
    try:
        profile = device_metadata.current()
        assert profile['os'] == 'Linux (WSL)'
        assert profile['machine_id'] == 'b' * 16
        assert profile['architecture'] == 'x86_64'
        assert profile['observed_at'] > 0
    finally:
        device_metadata.current.cache_clear()
