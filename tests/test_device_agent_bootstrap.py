"""Paired-device agent recovery preserves scope and denies revoked trust."""

import base64
import concurrent.futures
import json
import threading
from types import SimpleNamespace

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.fleet import device_auth
from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.onboard.requests import Device
from ml_stack.fleet.onboard.web import Call
from ml_stack.fleet.remote import Peer
from ml_stack.http import Server, ServerError
from ml_stack.workspace import (
    coordinator,
    coordinator_client,
    coordinator_config,
    device_sessions,
    tokens,
)
from ml_stack.workspace.coordination import workspace_id
from ml_stack.workspace.coordinator_client import Remote
from ml_stack.workspace.identity import AGENT, Denied
from ml_stack.workspace.remote_host import WorkspaceHost


@pytest.fixture
def enrolled(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    device = Device('a' * 64, 'test-device', 'test-host', '127.0.0.1', 1,
                    mine=True, secret=base64.urlsafe_b64encode(b'x' * 32).decode())
    projects = SimpleNamespace(machine='authority', hosts=lambda host: host == 'https://authority:8770', list=lambda: [
        {'id': 'b' * 32, 'name': 'example', 'board_host': 'https://authority:8770'}])
    document = {'name': 'codex', 'model': 'example-model', 'harness': 'codex',
                'project': {'key': 'b' * 32, 'name': 'example'}}
    return kit, device, projects, document


def test_lost_and_expired_session_keep_identity_project_and_model(enrolled):
    kit, device, projects, document = enrolled
    name, first = device_sessions.ensure(kit.ws, device, projects, document)
    info = kit.ws.registry.info(name)
    assert info['role'] == AGENT and info['project']['key'] == 'b' * 32
    assert info['model_state'] == 'claimed'
    assert device_sessions.ensure(kit.ws, device, projects, document, first) == (name, first)
    tokens.directory(kit.base).joinpath(name).unlink()
    recovered_name, recovered = device_sessions.ensure(kit.ws, device, projects, document)
    assert recovered_name == name and recovered != first
    with pytest.raises(Denied):
        kit.ws.auth(first)
    kit.ws.clock = lambda: info['expires'] + 1
    kit.ws.registry.clock = kit.ws.clock
    assert device_sessions.ensure(kit.ws, device, projects, document, recovered)[0] == name
    assert kit.ws.registry.info(name)['project'] == info['project']


def test_revoked_device_and_identity_cannot_recover(enrolled):
    kit, device, projects, document = enrolled
    name, token = device_sessions.ensure(kit.ws, device, projects, document)
    device.status = 'revoked'
    assert device_auth.identify([device], device_auth.secret(device)) is None
    with pytest.raises(Denied):
        device_sessions.ensure(kit.ws, device, projects, document, token)
    with pytest.raises(Denied):
        device_sessions.check(kit.ws, token, device, projects)
    device.status = 'active'
    kit.ws.revoke(kit.owner, name)
    with pytest.raises(Denied):
        device_sessions.ensure(kit.ws, device, projects, document)


def test_unowned_device_needs_existing_binding_and_foreign_project_denied(enrolled):
    kit, device, projects, document = enrolled
    device.mine = False
    with pytest.raises(Denied):
        device_sessions.ensure(kit.ws, device, projects, document)
    device.mine = True
    with pytest.raises(Denied):
        device_sessions.ensure(kit.ws, device, projects, {**document, 'project': {'name': 'foreign'}})
    name, token = device_sessions.ensure(kit.ws, device, projects, document)
    device.mine = False
    assert device_sessions.ensure(kit.ws, device, projects, document, token)[0] == name


def test_two_devices_cannot_recover_each_others_identity(enrolled):
    kit, device, projects, document = enrolled
    first_name, first = device_sessions.ensure(kit.ws, device, projects, document)
    second_device = Device('c' * 64, 'second', 'second-host', '127.0.0.2', 1, mine=True)
    second_name, second = device_sessions.ensure(kit.ws, second_device, projects, document)
    assert first_name != second_name
    with pytest.raises(Denied):
        device_sessions.ensure(kit.ws, second_device, projects, document, first)
    with pytest.raises(Denied):
        device_sessions.check(kit.ws, second, device, projects)


def test_bootstrap_route_requires_device_proof_and_existing_project(enrolled):
    kit, device, projects, document = enrolled
    identity = workspace_id(kit.ws)
    coordinator_config.save(kit.base, {'mode': 'host', 'workspace': identity})
    body = json.dumps({'workspace': identity, **document}).encode()
    call = Call('POST', '/workspace/v1/ensure', {}, '127.0.0.1', True, lambda most: body)
    assert coordinator.answer(kit.ws, call, projects=projects)[0] == 403
    status, result = coordinator.answer(kit.ws, call, device=device, projects=projects)
    assert status == 200 and kit.ws.auth(result['token']).role == AGENT
    device.status = 'revoked'
    assert coordinator.answer(kit.ws, call, device=device, projects=projects)[0] == 403


def test_http_device_session_recovery_and_revocation(enrolled, tmp_path, monkeypatch):
    kit, device, projects, document = enrolled
    identity = workspace_id(kit.ws)
    coordinator_config.save(kit.base, {'mode': 'host', 'workspace': identity})
    monkeypatch.setattr('ml_stack.fleet.api.entry_points', lambda **kwargs: [
        SimpleNamespace(name='workspace', load=lambda: coordinator.route)])
    runner = JobRunner(tmp_path / 'runner')
    server = Server(('127.0.0.1', 0), make_handler(Daemon(
        runner, tmp_path / 'files', 'legacy-cluster', projects=projects, devices=lambda: [device])))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    peer = Peer(f'http://127.0.0.1:{server.server_address[1]}', device_auth.secret(device))
    remote = Remote({'workspace': identity}, peer)
    base = tmp_path / 'other-device'
    base.mkdir()
    try:
        name = remote.ensure(base, 'codex', project=document['project'])
        token = tokens.load(base, name)
        assert remote.command(['whoami'], token)['id'] == name
        tokens.directory(base).joinpath(name).unlink()
        assert remote.ensure(base, 'codex', project=document['project']) == name
        recovered = tokens.load(base, name)
        assert recovered != token
        device.status = 'revoked'
        with pytest.raises(ServerError):
            remote.command(['whoami'], recovered)
        with pytest.raises(ServerError):
            remote.ensure(base, 'codex', project=document['project'])
    finally:
        server.shutdown()
        server.server_close()
        runner.shutdown()


def test_project_bootstrap_has_same_device_boundary(enrolled):
    kit, device, projects, document = enrolled
    projects.workspace_base = lambda project_id: kit.base
    host = WorkspaceHost(projects)
    body = {**document, 'agent_token': '', 'device': {}}
    project_id = document['project']['key']
    assert host.answer(project_id, 'ensure', body)[0] == 403
    status, result = host.answer(project_id, 'ensure', body, device=device)
    assert status == 200
    request = {'agent_token': result['token'], 'operation': 'whoami', 'args': [], 'kwargs': {}}
    assert host.answer(project_id, 'board', request, device=device)[0] == 200
    assert host.answer(project_id, 'board', request)[0] == 403


def test_valid_session_records_new_claimed_model_and_harness(enrolled):
    kit, device, projects, document = enrolled
    name, token = device_sessions.ensure(kit.ws, device, projects, document)
    claim = {**document, 'model': 'another-model', 'harness': 'claude-code'}
    assert device_sessions.ensure(kit.ws, device, projects, claim, token) == (name, token)
    info = kit.ws.registry.info(name)
    assert (info['model'], info['harness'], info['model_state']) == (
        'another-model', 'claude-code', 'claimed')
    assert [row['model'] for row in info['models']] == ['example-model', 'another-model']
    assert not any(row['verified'] for row in info['models'])


@pytest.mark.parametrize('credential', ['', 'valid', 'expired'])
def test_bound_session_refuses_different_authoritative_project(enrolled, credential):
    kit, device, projects, document = enrolled
    name, token = device_sessions.ensure(kit.ws, device, projects, document)
    other = {'id': 'c' * 32, 'name': 'other', 'board_host': 'https://authority:8770'}
    initial = projects.list()
    projects.list = lambda: [*initial, other]
    requested = {**document, 'project': {'key': other['id'], 'name': 'example'}}
    presented = token if credential == 'valid' else 'expired-session' if credential else ''
    with pytest.raises(Denied, match='requested project differs'):
        device_sessions.ensure(kit.ws, device, projects, requested, presented)
    assert kit.ws.auth(token).id == name
    assert kit.ws.registry.info(name)['project'] == document['project']


def test_concurrent_recovery_keeps_the_persisted_session_valid(enrolled, tmp_path):
    kit, device, projects, document = enrolled
    base = tmp_path / 'client'
    base.mkdir()
    first_issued, second_entered, release = threading.Event(), threading.Event(), threading.Event()
    counter, counter_lock = [0], threading.Lock()

    def request(method, path, *, data, headers):
        incoming = json.loads(data)
        name, token = device_sessions.ensure(
            kit.ws, device, projects, incoming, headers['X-ML-Stack-Workspace-Token'])
        with counter_lock:
            counter[0] += 1
            number = counter[0]
        if number == 1:
            first_issued.set()
            assert release.wait(5)
        else:
            second_entered.set()
        return 200, json.dumps({'workspace': 'workspace:test', 'agent': name, 'token': token}).encode(), {}

    remote = Remote({'workspace': 'workspace:test'}, SimpleNamespace(_request=request))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(remote.ensure, base, 'codex', project=document['project'])
        assert first_issued.wait(5)
        second = workers.submit(remote.ensure, base, 'codex', project=document['project'])
        try:
            assert not second_entered.wait(0.2)
        finally:
            release.set()
        first_name, second_name = first.result(timeout=5), second.result(timeout=5)
    assert first_name == second_name
    assert kit.ws.auth(tokens.load(base, first_name)).id == first_name


@pytest.mark.parametrize('count', [0, 2])
def test_enrolled_device_never_creates_local_authority_without_unique_coordinator(tmp_path, monkeypatch, count):
    base = tmp_path / 'workspace'
    monkeypatch.setattr(coordinator_client, 'load_cluster_key', lambda: b'enrolled')
    monkeypatch.setattr(coordinator_client, 'discover', lambda: [(None, {})] * count)
    if count:
        with pytest.raises(Denied, match='ambiguous'):
            coordinator_client.client(base)
    else:
        assert coordinator_client.client(base) is None
    assert not (base / 'agents.json').exists()
    assert not coordinator_config.load(base)


@pytest.mark.parametrize('surface', ['coordinator', 'board'])
@pytest.mark.parametrize('change', ['unshared', 'other-authority'])
def test_existing_session_loses_access_when_hosted_project_authorization_ends(enrolled, surface, change):
    kit, device, projects, document = enrolled
    name, token = device_sessions.ensure(kit.ws, device, projects, document)
    identity = workspace_id(kit.ws)
    coordinator_config.save(kit.base, {'mode': 'host', 'workspace': identity})
    projects.workspace_base = lambda project_id: kit.base
    host = WorkspaceHost(projects)

    def invoke():
        if surface == 'board':
            return host.answer(document['project']['key'], 'board', {
                'agent_token': token, 'operation': 'whoami', 'args': [], 'kwargs': {}}, device=device)
        body = json.dumps({'workspace': identity, 'request_id': 'c' * 32, 'argv': ['whoami']}).encode()
        call = Call('POST', '/workspace/v1/call', {'X-ML-Stack-Workspace-Token': token},
                    '127.0.0.1', True, lambda most: body)
        return coordinator.answer(kit.ws, call, device=device, projects=projects)

    assert invoke()[0] == 200
    initial = projects.list()
    projects.list = lambda: [] if change == 'unshared' else [{**initial[0], 'board_host': 'https://elsewhere:8770'}]
    assert kit.ws.auth(token).id == name
    assert invoke()[0] == 403


def test_delegated_session_inherits_root_device_and_live_project_boundary(enrolled):
    kit, device, projects, document = enrolled
    _name, parent = device_sessions.ensure(kit.ws, device, projects, document)
    child = kit.ws.delegate(parent, 'helper')
    token = tokens.load(kit.base, child['id'])
    device_sessions.check(kit.ws, token, device, projects)
    with pytest.raises(Denied, match='active paired device'):
        device_sessions.check(kit.ws, token, None, projects)
    projects.list = lambda: []
    with pytest.raises(Denied, match='not uniquely hosted'):
        device_sessions.check(kit.ws, token, device, projects)


@pytest.mark.parametrize('change', ['revoked', 'expired', 'child-project'])
def test_delegated_session_cannot_outlive_or_widen_root_authorization(enrolled, change):
    kit, device, projects, document = enrolled
    name, parent = device_sessions.ensure(kit.ws, device, projects, document)
    child = kit.ws.delegate(parent, 'helper')
    token = tokens.load(kit.base, child['id'])
    agents = kit.ws.registry._load()
    if change == 'revoked':
        agents[name]['revoked'] = True
    elif change == 'expired':
        agents[name]['expires'] = kit.ws.clock() - 1
    else:
        agents[child['id']]['project'] = {'key': 'f' * 32, 'name': 'foreign'}
    kit.ws.registry._save(agents)
    with pytest.raises(Denied):
        device_sessions.check(kit.ws, token, device, projects)


@pytest.fixture
def saved_project_actor(tmp_path, monkeypatch):
    from ml_stack.workspace import device_agent, onboard, project

    monkeypatch.setattr(device_agent, 'device_id', lambda: 'abcdef0123456789')
    root = tmp_path / 'project'
    root.mkdir()
    kit = Kit(clean_env(monkeypatch, tmp_path))
    token = kit.ws.registry.mint(onboard.SETUP, 'worker', AGENT, 3600)
    kit.ws.registry.set_project(kit.ws.auth(kit.owner), 'worker', project.describe(str(root)))
    tokens.store(kit.base, 'worker', token)
    return kit, root, token


def test_saved_setup_standard_agent_has_project_access_without_worker_device_rights(saved_project_actor):
    from ml_stack.workspace import device_agent

    kit, root, token = saved_project_actor
    before = kit.ws.registry.path.read_bytes()
    with device_agent.owned_project_session(kit.ws, token, 'worker', root) as actor:
        assert actor.id == 'worker'
    assert kit.ws.registry.path.read_bytes() == before
    assert not (kit.base / 'device-accounts.db').exists()
    with pytest.raises(Denied, match='trusted local device registration'):
        device_agent.owned_local(kit.ws, token)


@pytest.mark.redteam
@pytest.mark.parametrize('attack', ['revoked', 'expired', 'copied', 'id', 'symlink', 'owner',
                                  'child', 'lead', 'person', 'device', 'project'])
def test_saved_standard_project_session_preserves_identity_scope_and_storage(saved_project_actor, tmp_path, monkeypatch, attack):
    from ml_stack.workspace import device_agent, project

    kit, root, token = saved_project_actor
    requested = 'worker'
    entries = kit.ws.registry._load()
    entry = entries['worker']
    if attack == 'revoked':
        entry['revoked'] = True
    elif attack == 'expired':
        entry['expires'] = 1
    elif attack == 'lead':
        entry['role'] = 'lead'
    elif attack == 'person':
        entry['role'] = 'human'
    elif attack == 'device':
        entry['session_device'] = 'e' * 64
    elif attack == 'project':
        entry['project'] = project.describe(str(tmp_path))
    kit.ws.registry._save(entries)
    if attack == 'child':
        delegated = kit.ws.delegate(token, 'child')
        requested = delegated['id']
        token = tokens.load(kit.base, requested)
    elif attack == 'copied':
        foreign = Kit(tmp_path / 'foreign')
        token = foreign.agent('worker')
        tokens.store(kit.base, 'worker', token)
    elif attack == 'id':
        requested = 'other'
    elif attack == 'symlink':
        saved = tokens.directory(kit.base) / 'worker'
        copied = tmp_path / 'copied-token'
        copied.write_text(token)
        copied.chmod(0o600)
        saved.unlink()
        saved.symlink_to(copied)
    elif attack == 'owner':
        real = tokens.problem
        monkeypatch.setattr(tokens, 'problem', lambda path: 'belongs to another user'
                            if path == kit.ws.registry.path else real(path))
    before = kit.ws.registry.path.read_bytes()
    with pytest.raises(Denied), device_agent.owned_project_session(kit.ws, token, requested, root):
        pytest.fail('invalid saved actor received project attachment authority')
    assert kit.ws.registry.path.read_bytes() == before


def test_saved_project_scope_accepts_its_primary_worktree_equivalent(saved_project_actor, tmp_path, monkeypatch):
    from ml_stack.workspace import device_agent

    kit, primary, token = saved_project_actor
    checkout = tmp_path / 'worktree'
    checkout.mkdir()
    monkeypatch.setattr(device_agent.worktreerules, 'checkouts', lambda path: (checkout, primary))
    with device_agent.owned_project_session(kit.ws, token, 'worker', checkout) as actor:
        assert actor.id == 'worker'


def test_registration_records_reported_device_and_authenticated_peer_then_refreshes(enrolled):
    kit, device, projects, document = enrolled
    metadata = {'machine_id': '1' * 16, 'device_id': 'b' * 64,
                'hostname': 'worker', 'os': 'Linux', 'architecture': 'x86_64',
                'runtime_version': '0.2.2', 'runtime_commit': 'c' * 40,
                'peer_id': 'f' * 64, 'verification': 'local-observed'}
    name, token = device_sessions.ensure(kit.ws, device, projects, {**document, 'device': metadata})
    before = kit.ws.auth(token)
    profile = kit.ws.registry.info(name)['device']
    assert profile['os'] == 'Linux' and profile['hostname'] == 'worker'
    assert profile['machine_id'] is None and profile['device_id'] is None
    assert profile['peer_id'] == device.fingerprint
    assert profile['peer_verification'] == 'paired'
    assert profile['verification'] == 'agent-reported'
    child = kit.ws.delegate(token, 'helper')
    inherited = kit.ws.registry.info(child['id'])['device']
    assert (inherited['verification'], inherited['inherited_from'], inherited['parent_verification']) == (
        'inherited', name, 'agent-reported')
    assert {key: inherited[key] for key in ('hostname', 'os', 'peer_id', 'peer_verification')} == {
        key: profile[key] for key in ('hostname', 'os', 'peer_id', 'peer_verification')}
    updated = {**metadata, 'runtime_version': '0.2.3', 'runtime_commit': 'd' * 40}
    assert device_sessions.ensure(kit.ws, device, projects, {**document, 'device': updated}, token) == (name, token)
    assert kit.ws.registry.info(name)['device']['runtime_commit'] == 'd' * 40
    assert kit.ws.auth(token) == before


@pytest.mark.redteam
def test_malformed_registration_device_cannot_create_identity(enrolled):
    kit, device, projects, document = enrolled
    before = kit.ws.registry.ids()
    with pytest.raises(ValueError):
        device_sessions.ensure(kit.ws, device, projects, {**document, 'device': {'machine_id': 'invented'}})
    assert kit.ws.registry.ids() == before


@pytest.mark.parametrize('mode', ['dev', 'paired'])
def test_cached_delegated_token_returns_without_root_registration(enrolled, mode):
    from ml_stack.workspace.remote import RemoteWorkspace
    kit, device, projects, document = enrolled
    _name, parent = device_sessions.ensure(kit.ws, device, projects, document)
    child = kit.ws.delegate(parent, 'helper')
    cached = tokens.load(kit.base, child['id'])
    remote = object.__new__(RemoteWorkspace)
    remote.base, remote.mode = kit.base, mode
    remote.call = lambda operation, token: kit.ws.auth(token)
    def refused(*args, **kwargs):
        pytest.fail('a delegated token must not register or renew its root')
    remote._request = refused
    assert remote.token(agent=child['id']) == cached
