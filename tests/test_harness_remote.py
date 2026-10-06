"""Canonical native ownership keeps physical claims and cleanup local."""

from types import SimpleNamespace

import pytest

from ml_stack import harness_remote, harnesshook
from ml_stack.workspace import integration_git as repo, worktree_lifecycle
from ml_stack.workspace.claims import Conflict
from ml_stack.workspace.identity import AGENT, Denied, Identity


@pytest.fixture
def setup(tmp_path, monkeypatch):
    primary = tmp_path / 'repository'
    primary.mkdir()
    repo.git(primary, 'init', '-b', 'development')
    repo.git(primary, 'config', 'user.name', 'Fixture')
    repo.git(primary, 'config', 'user.email', 'fixture@example.test')
    (primary / 'source.py').write_text('value = 1\n')
    repo.git(primary, 'add', 'source.py')
    repo.git(primary, 'commit', '-m', 'chore: fixture')
    checkout = tmp_path / 'checkout'
    repo.git(primary, 'worktree', 'add', '-b', 'worker/change', str(checkout))
    info = {'id': 'worker', 'role': AGENT, 'parent': '', 'can': ['read', 'claim'],
            'project': {'key': 'a' * 32}}
    binding = {'host': 'http://127.0.0.1:9', 'project_id': 'a' * 32, 'agent': 'worker'}
    remote = SimpleNamespace(host=binding['host'], project_id=binding['project_id'], base=tmp_path / 'remote',
                             token=lambda **kwargs: 'fixture', call=lambda *args: info,
                             native_reserve=lambda *args: [])
    monkeypatch.setattr(harness_remote.project_connection, 'selected', lambda root=None: binding)
    monkeypatch.setattr(harness_remote.project_connection, 'RemoteWorkspace', lambda *args, **kwargs: remote)
    monkeypatch.setattr(harness_remote.limits, 'root', lambda: tmp_path / 'physical')
    return SimpleNamespace(primary=primary, checkout=checkout, remote=remote, info=info, binding=binding,
                           who=Identity('worker', AGENT, can=('read', 'claim')))


@pytest.mark.redteam
@pytest.mark.parametrize('change', [
    {'id': 'foreign'}, {'role': 'human'}, {'project': {'key': 'b' * 32}}, {'can': ['read']},
])
def test_canonical_identity_and_mutation_permission_are_authenticated(setup, change):
    setup.info.update(change)
    with pytest.raises(Denied):
        harness_remote.context('worker', str(setup.checkout), [str(setup.checkout)])
    assert not harness_remote.claims().path.exists()


@pytest.mark.redteam
def test_remote_authentication_failure_has_no_local_credential_fallback(setup):
    def refused(*args):
        raise Denied('fixture expired capability')
    setup.remote.call = refused
    with pytest.raises(Denied, match='expired capability'):
        harness_remote.context('worker', str(setup.checkout), [str(setup.checkout)])
    assert not harness_remote.claims().path.exists()


@pytest.mark.redteam
def test_physical_claims_conflict_across_canonical_authorities(setup):
    target = str(setup.checkout / 'source.py')
    harness_remote.reserve(setup.remote, setup.who, [('file', target)], {})
    setup.remote.project_id = 'b' * 32
    setup.binding['project_id'] = setup.remote.project_id
    with pytest.raises(Conflict):
        harness_remote.reserve(setup.remote, setup.who, [('file', target)], {})


@pytest.mark.redteam
def test_remote_refusal_rolls_back_only_new_physical_claims(setup):
    store = harness_remote.claims()
    principal = harness_remote.physical_owner(setup.remote, setup.who)
    old, new = str(setup.checkout / 'source.py'), str(setup.checkout / 'new.py')
    store.reserve(principal, [('file', old)])
    def refused(*args):
        raise Denied('fixture remote claim refused')
    setup.remote.native_reserve = refused
    with pytest.raises(Denied, match='remote claim refused'):
        harness_remote.reserve(setup.remote, setup.who, [('file', old), ('file', new)], {})
    assert store.who('file', old)['owner'] == principal.id
    assert store.who('file', new) is None


@pytest.mark.redteam
def test_concurrent_changed_claim_survives_rollback(setup):
    store = harness_remote.claims()
    principal = harness_remote.physical_owner(setup.remote, setup.who)
    target = str(setup.checkout / 'source.py')
    def changed(*args):
        store.reserve(principal, [('file', target)], {'note': 'fixture later reservation'})
        raise Denied('fixture remote claim refused')
    setup.remote.native_reserve = changed
    with pytest.raises(Denied):
        harness_remote.reserve(setup.remote, setup.who, [('file', target)], {})
    assert store.who('file', target)['note'] == 'fixture later reservation'


@pytest.mark.redteam
def test_actual_target_must_belong_to_canonical_binding(setup, monkeypatch):
    monkeypatch.setattr(harness_remote.project_connection, 'selected',
                        lambda root: {**setup.binding, 'project_id': 'b' * 32})
    with pytest.raises(Denied, match='another canonical project'):
        harness_remote.reserve(setup.remote, setup.who, [('file', str(setup.checkout / 'source.py'))], {})
    assert not harness_remote.claims().path.exists()


@pytest.mark.redteam
def test_assigned_physical_claims_keep_assignment_authorization(setup):
    store = harness_remote.claims()
    principal = harness_remote.physical_owner(setup.remote, setup.who)
    target = str(setup.checkout)
    store.reserve(principal, [('worktree', target)], {'assignment': 'fixture-task'})
    with pytest.raises(Denied, match='different task assignment'):
        harness_remote.reserve(setup.remote, setup.who, [('worktree', target)], {}, branch_only=True)
    assert store.who('worktree', target)['assignment'] == 'fixture-task'


def test_reservation_reports_prior_keys_under_ownership_lock(setup):
    store = harness_remote.claims()
    principal = harness_remote.physical_owner(setup.remote, setup.who)
    old, new = str(setup.checkout / 'source.py'), str(setup.checkout / 'new.py')
    store.reserve(principal, [('file', old)])
    made, previous = store.reserve(principal, [('file', old), ('file', new)], with_previous=True)
    assert previous == {f'file:{old}'}
    assert {row['key'] for row in made} == {old, new}


@pytest.mark.redteam
def test_stop_uses_canonical_identity_for_local_durable_cleanup(setup):
    harness_remote.reserve(setup.remote, setup.who, [('file', str(setup.checkout / 'source.py'))], {})
    assert worktree_lifecycle.pending(setup.remote.base, 'worker')[0]['path'] == str(setup.checkout)
    rail = harnesshook.Rail('plan-and-go', 'worker', roots=[str(setup.checkout)])
    assert harnesshook.stop(rail)['decision'] == 'block'
    repo.git(setup.primary, 'worktree', 'remove', str(setup.checkout))
    repo.git(setup.primary, 'branch', '-d', 'worker/change')
    setup.info['can'] = ['read']
    assert harnesshook.stop(rail) == {}


@pytest.mark.redteam
def test_native_reservation_uses_actual_remote_capability_api(setup, monkeypatch):
    from ml_stack.workspace import tokens
    from ml_stack.workspace.remote import RemoteWorkspace

    remote = RemoteWorkspace.__new__(RemoteWorkspace)
    remote.host, remote.project_id, remote.base = setup.remote.host, setup.remote.project_id, setup.remote.base
    remote.cluster = 'fixture-cluster'
    tokens.store(remote.base, 'worker', 'fixture-agent-capability')
    requests = []
    def request(action, payload):
        requests.append((action, payload))
        return {'result': setup.info if payload['operation'] == 'whoami' else []}
    monkeypatch.setattr(remote, '_request', request)
    monkeypatch.setattr(harness_remote.project_connection, 'RemoteWorkspace', lambda *args, **kwargs: remote)
    authenticated, who = harness_remote.context('worker', str(setup.checkout), [str(setup.checkout)])
    harness_remote.reserve(authenticated, who, [('file', str(setup.checkout / 'source.py'))], {})
    operation = requests[-1][1]
    assert operation['operation'] == 'native.reserve'
    assert operation['args'] == [[('area', 'source.py')]]
    assert operation['agent_token'] == 'fixture-agent-capability'


@pytest.mark.parametrize('command, expected', [
    ('git add source.py', True), ('git -C /project commit -m fixture', True),
    ('git add source.py && git commit -m fixture', True), ('git checkout development', False),
    ('git add source.py; python mutate.py', False), ('git add source.py > changed.txt', False),
    ('git -c core.worktree=/outside add source.py', False), ('git -C /root -C ../outside add source.py', False),
    ('/tmp/untrusted/git add source.py', False), ('./git add source.py', False),
    ('git add $SOURCE', False), ('git add *.py', False), ('printf fixture | git add source.py', False),
])
def test_only_staging_and_commit_shell_commands_use_branch_claims(command, expected):
    assert harness_remote.staging_only('Bash', {'command': command}) is expected


def test_staging_claims_exact_branch_without_project_wide_area(setup):
    calls = []
    setup.remote.native_reserve = lambda actor, resources: calls.append((actor, resources))
    harness_remote.reserve(setup.remote, setup.who, [('worktree', str(setup.checkout))], {}, branch_only=True)
    assert calls == [('worker', [('branch', 'worker/change')])]


@pytest.mark.redteam
def test_unknown_root_mutation_requires_bounded_source_targets(setup):
    with pytest.raises(Denied, match='bounded source paths'):
        harness_remote.reserve(setup.remote, setup.who, [('worktree', str(setup.checkout))], {})
    assert not harness_remote.claims().path.exists()


@pytest.mark.redteam
def test_revoked_physical_claim_cleanup_preserves_other_identities(setup):
    store = harness_remote.claims()
    mine, other = str(setup.checkout / 'source.py'), str(setup.checkout / 'other.py')
    store.reserve(harness_remote.physical_owner(setup.remote, setup.who), [('file', mine)])
    store.reserve(Identity('foreign', AGENT), [('file', other)])
    snapshot = harness_remote.revocation_snapshot(setup.remote, 'worker')
    harness_remote.release_revoked(setup.remote, 'worker', snapshot)
    assert store.who('file', mine) is None
    assert store.who('file', other)['owner'] == 'foreign'


@pytest.mark.redteam
def test_recreated_same_identity_claim_survives_revocation_snapshot(setup):
    target = str(setup.checkout / 'source.py')
    harness_remote.reserve(setup.remote, setup.who, [('file', target)], {})
    snapshot = harness_remote.revocation_snapshot(setup.remote, 'worker')
    harness_remote.reserve(setup.remote, setup.who, [('file', target)], {})
    harness_remote.release_revoked(setup.remote, 'worker', snapshot)
    assert harness_remote.claims().who('file', target) is not None


@pytest.mark.redteam
def test_inherited_git_location_overrides_cannot_use_branch_only_admission(monkeypatch):
    monkeypatch.setenv('GIT_WORK_TREE', '/fixture/outside')
    assert harness_remote.staging_only('Bash', {'command': 'git add source.py'}) is False


@pytest.mark.redteam
def test_unknown_nested_worktree_mutation_cannot_claim_the_wrong_area(setup):
    nested = setup.checkout / 'nested'
    nested.mkdir()
    with pytest.raises(Denied, match='bounded source paths'):
        harness_remote.reserve(setup.remote, setup.who, [('worktree', str(nested))], {})
    assert not harness_remote.claims().path.exists()


def test_local_area_owner_blocks_canonical_file_reservation(setup):
    target = str(setup.checkout / 'source.py')
    harness_remote.claims().reserve(Identity('other', AGENT), [('area', target)])
    with pytest.raises(Conflict):
        harness_remote.reserve(setup.remote, setup.who, [('file', target)], {})


@pytest.mark.parametrize('command', ['echo x > "$TARGET"', 'touch *.py', 'touch $(pwd)/x'])
def test_canonical_shell_targets_must_be_fixed(command):
    with pytest.raises(Denied):
        harness_remote.inspect_shell('Bash', {'command': command})


def cli_args(**changes):
    return SimpleNamespace(**{'cmd': 'claim', 'agent': 'worker', 'label': '', 'kind': 'file',
                              'key': '', 'ttl': 0, 'pid': 0, 'note': '', **changes})


@pytest.mark.parametrize('change', [{'agent': 'foreign'}, {'kind': 'install'}, {'ttl': 5}, {'pid': 1}])
def test_canonical_cli_refuses_unbound_or_unsupported_claims(setup, change):
    args = cli_args(key=str(setup.checkout / 'source.py'), **change)
    with pytest.raises(Denied):
        harness_remote.cli_command(setup.remote, 'fixture', args)


def test_canonical_cli_claim_maps_bounded_source_and_physical_area(setup):
    recorded = []
    setup.remote.native_reserve = lambda name, resources: recorded.extend(resources)
    target = str(setup.checkout / 'source.py')
    harness_remote.cli_command(setup.remote, 'fixture', cli_args(key=target))
    assert recorded == [('area', 'source.py')]
    assert harness_remote.claims().who('area', target)['owner'] == harness_remote.physical_owner(setup.remote, setup.who).id


def test_canonical_cli_heartbeat_preserves_requested_ttl(setup):
    calls = []
    def call(operation, token, **kwargs):
        calls.append((operation, kwargs))
        return setup.info if operation == 'whoami' else []
    setup.remote.call = call
    result = harness_remote.cli_command(setup.remote, 'fixture', cli_args(cmd='heartbeat', ttl=17))
    assert ('native.heartbeat', {'ttl_s': 17}) in calls
    assert result == {'renewed': 0, 'capped': []}


def test_canonical_cli_release_exact_child_preserves_covering_parent(setup):
    store = harness_remote.claims()
    principal = harness_remote.physical_owner(setup.remote, setup.who)
    parent, child = str(setup.checkout), str(setup.checkout / 'source.py')
    store.reserve(principal, [('worktree', parent), ('file', child), ('area', child)])
    setup.remote.native_release = lambda *args: {'released': True}
    harness_remote.cli_command(setup.remote, 'fixture', cli_args(cmd='release', key=child))
    rows = store.listing()
    assert [(row['kind'], row['key']) for row in rows] == [('worktree', parent)]


def test_canonical_cli_release_preserves_recreated_exact_claim(setup):
    target = str(setup.checkout / 'source.py')
    harness_remote.reserve(setup.remote, setup.who, [('file', target)], {})
    def release(*args):
        harness_remote.reserve(setup.remote, setup.who, [('file', target)], {})
        return {'released': True}
    setup.remote.native_release = release
    harness_remote.cli_command(setup.remote, 'fixture', cli_args(cmd='release', key=target))
    assert harness_remote.claims().who('file', target)
    assert harness_remote.claims().who('area', target)
