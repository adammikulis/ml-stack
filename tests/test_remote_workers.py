"""Dev worker launches retain local runtime state and one sealed Board."""

import base64
import hashlib
import threading
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from ml_stack import http
from ml_stack.fleet import api, discovery, project_enrollment, tls
from ml_stack.fleet.discovery import (
    Beacon,
    Membership,
    _write_memberships,
    derive_token,
    memberships,
)
from ml_stack.fleet.framing import LimitedServer
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.remote import Peer
from ml_stack.workspace import guide, localagent, localloop, localmodel, remote_workers, tokens
from ml_stack.workspace.remote import RemoteWorkspace
from ml_stack.workspace.remote_host import WorkspaceHost
from ml_stack.workspace.service import Workspace

PROJECT = 'a' * 32
CLUSTER = 'development'
KEY = base64.urlsafe_b64encode(bytes(range(32))).rstrip(b"=")
CLUSTER_ID = hashlib.sha256(KEY).hexdigest()


def _server(root, machine, projects):
    files = root / machine / 'files'
    files.mkdir()
    runner = JobRunner(root / machine / 'jobs', files)
    daemon = api.Daemon(runner, files, lambda: derive_token(memberships()[0].key))
    daemon.projects, daemon.workspaces = projects, WorkspaceHost(projects)
    ident = tls.identity(root / machine / 'tls', machine)
    server = LimitedServer((discovery.primary_ip(), 0), api.make_handler(daemon), tls=tls.server_context(ident))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    beacon = Beacon(name=machine, machine=machine, host=discovery.primary_ip(), port=server.server_port, cert=ident.beacon)
    peer = Peer(beacon.base_url, derive_token(KEY), beacon=beacon)
    http.pin(urlsplit(beacon.base_url).netloc, tls.pinned_context(ident.beacon))
    return daemon, server, runner, peer


def _membership(root):
    keyfile = root / 'cluster.key'
    keyfile.write_bytes(KEY)
    keyfile.chmod(0o600)
    _write_memberships([Membership(CLUSTER, KEY, mode='dev')], keyfile)
    return keyfile


@pytest.fixture
def devices(tmp_path, monkeypatch):
    monkeypatch.setenv('ML_STACK_CLUSTER_KEY', str(_membership(tmp_path)))
    monkeypatch.setattr('ml_stack.home.state', lambda *parts: tmp_path / 'device-a' / 'state' / '/'.join(parts))
    monkeypatch.setattr('ml_stack.home.device_id', lambda: 'device-b')
    monkeypatch.setattr('ml_stack.home.machine_id', lambda: remote_workers.home.device_id())
    monkeypatch.setattr(guide.coordinator_bootstrap, 'ensure_host', lambda *args: None)
    roots, servers, runners, peers, spawned = {}, [], [], [], []
    project = SimpleNamespace(name='sample', board_host='', root='')
    class Projects:
        def __init__(self, machine):
            self.machine = machine
        def hosts(self, board_host):
            return bool(board_host) and board_host == project.board_host and self.machine == 'device-a'
        def get(self, ident):
            if ident != PROJECT:
                raise ValueError('unknown project')
            return SimpleNamespace(**{**vars(project), 'root': str(roots[self.machine]), 'id': ident})
        def workspace_base(self, ident):
            self.get(ident)
            return tmp_path / self.machine / 'canonical' / ident
    try:
        for machine in ('device-a', 'device-b'):
            root = tmp_path / machine / 'checkout'
            root.mkdir(parents=True)
            roots[machine] = root
            daemon, server, runner, node = _server(tmp_path, machine, Projects(machine))
            runners.append(runner)
            servers.append(server)
            peers.append(node)
            beacon = node.beacon
            if machine == 'device-a':
                project.board_host = beacon.base_url
                authority = daemon.workspaces
                authority.prepare(PROJECT)
            else:
                target_projects = daemon.projects
        monkeypatch.setattr(Peer, 'discover', lambda **kwargs: peers)
        actual_remote = RemoteWorkspace
        def target_remote(*args, **kwargs):
            result = actual_remote(*args, **kwargs)
            if (threading.current_thread() is not threading.main_thread() or
                    remote_workers.home.device_id() != 'device-a'):
                result.base = tmp_path / 'device-b' / 'remote-sessions'
            return result
        monkeypatch.setattr(remote_workers, 'RemoteWorkspace', target_remote)
        local = Workspace(tmp_path / 'device-b' / 'workspace')
        monkeypatch.setattr(remote_workers, 'Workspace', lambda path=None: Workspace(path) if path else local)
        pick = localmodel.Pick(ref='fixture.gguf', name='Qwen3.8-test', size_bytes=1)
        monkeypatch.setattr(localmodel, 'choose', lambda *args, **kwargs: pick)
        monkeypatch.setattr('ml_stack.workspace.localprofile.admit', lambda *args: ('', ''))
        monkeypatch.setattr(remote_workers.jobs, 'detach', lambda module, args, **kwargs:
                            spawned.append((module, args, kwargs)) or SimpleNamespace(pid=0, log=kwargs['log']))
        caller = actual_remote(peers[0].base_url, PROJECT, cluster=CLUSTER)
        enrolled = caller.enroll('test-caller', model='test-runtime', harness='test-harness')
        caller_token = tokens.load(caller.base, enrolled['id'])
        target = actual_remote(peers[1].base_url, PROJECT, cluster=CLUSTER)
        body = {'agent_token': caller_token, 'cluster': CLUSTER, 'cluster_id': CLUSTER_ID,
                'name': 'local-qwen', 'ctx': 32768, 'max_output_tokens': 1234}
        yield SimpleNamespace(caller=caller, token=caller_token, target=target, body=body,
                              local=local, authority=authority, target_projects=target_projects,
                              spawned=spawned, roots=roots, peers=peers, project=project, keyfile=tmp_path / 'cluster.key')
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for runner in runners:
            runner.shutdown()
        for peer in peers:
            http.pin(urlsplit(peer.base_url).netloc, None)


def test_tls_launch_and_local_loop_exchange_on_the_canonical_board(devices):
    state = devices
    result = state.target._request('worker', state.body)
    assert result['project_id'] == PROJECT
    assert result['identity'].startswith('lan-device-b-')
    assert state.spawned[0][0] == 'ml_stack.workspace.remote_workers'
    assert state.spawned[0][1] == [str(state.local.base), 'local-qwen']
    agent = localagent.load(state.local, 'local-qwen')
    assert agent.project == str(state.roots['device-b'])
    assert agent.max_output_tokens == 1234
    assert agent.orders_from == ('test-caller',)
    record = __import__('ml_stack.files', fromlist=['read_json']).read_json(
        localagent.folder(state.local) / 'local-qwen.remote.json', {})
    worker = remote_workers.BoardWorker(state.local, record)
    assert worker.base == state.local.base
    assert worker.auth(worker.worker_token(agent)).id == result['identity']
    state.caller.call('send', state.token, result['identity'], 'question', 'Report the project Board identity')
    cancel, executed = threading.Event(), []
    def execute(agent, row, why, stopped):
        executed.append(row)
        cancel.set()
        return 'answer', 'Worker replied from the canonical project Board', 1
    settings = localloop.Settings(cancel=cancel, execute=execute,
                                 serve=lambda agent: localloop.Held(None, {'id': 'test-lease'}))
    assert localloop.run(worker, 'local-qwen', settings) == 0
    assert len(executed) == 1
    replies = state.caller.call('inbox', state.token, ack=False, raw=True)
    assert any(row['from'] == result['identity'] and row['type'] == 'answer'
               and 'canonical project Board' in row['raw'] for row in replies)
    assert state.local.inbox(tokens.load(state.local.base, agent.identity), ack=False) == []


@pytest.mark.redteam
@pytest.mark.parametrize('field,value', [('project', '/tmp'), ('cwd', '/tmp'), ('env', {}),
                                        ('ctx', True), ('ctx', -1), ('max_output_tokens', 0)])
def test_remote_launch_refuses_paths_and_invalid_budgets(devices, field, value):
    with pytest.raises(PermissionError):
        devices.target._request('worker', {**devices.body, field: value})
    assert not devices.spawned


@pytest.mark.redteam
@pytest.mark.parametrize('mode', ['prod', 'invalid-cluster', 'invalid-agent', 'unsealed'])
def test_remote_launch_requires_dev_sealed_transport_and_project_capability(devices, monkeypatch, mode):
    body = dict(devices.body)
    if mode == 'prod':
        monkeypatch.setattr(project_enrollment, 'memberships', lambda path: [Membership(CLUSTER, KEY, mode='prod')])
    elif mode == 'invalid-cluster':
        body['cluster_id'] = 'b' * 64
    elif mode == 'invalid-agent':
        body['agent_token'] = 'mlws1.no-such-agent.invalid'
    if mode == 'unsealed':
        with pytest.raises(http.ServerError):
            http.request_json(devices.target.endpoint + '/worker', payload=body)
    else:
        with pytest.raises(PermissionError):
            devices.target._request('worker', body)
    assert not devices.spawned


def test_repeated_launch_reuses_owned_running_worker(devices, monkeypatch):
    first = devices.target._request('worker', devices.body)
    monkeypatch.setattr(localagent, 'alive', lambda agent: True)
    second = devices.target._request('worker', devices.body)
    assert second['already'] is True
    assert second['identity'] == first['identity']
    assert len(devices.spawned) == 1
    assert first['requested_by'] == second['requested_by'] == 'test-caller'
    assert first['requested_context'] == second['requested_context'] == 32768
    assert first['effective_limits'] == second['effective_limits'] == {
        'context': 32768, 'max_output_tokens': 1234,
        'task_caps': {'rounds': None, 'calls': None, 'steps': None, 'seconds': None}}
    changed = devices.target._request('worker', {**devices.body, 'ctx': 65536, 'max_output_tokens': 99,
                                              'task_caps': {'rounds': 2}})
    assert changed['effective_limits'] == first['effective_limits']
    assert changed['requested_context'] == 32768


def test_cli_discovers_target_without_a_saved_project_connection(devices, monkeypatch):
    state = devices
    monkeypatch.setattr(remote_workers, 'selected', lambda root: None)
    discovered = []
    def attach(root):
        discovered.append(root)
        return {'host': state.caller.host, 'project_id': PROJECT, 'cluster': CLUSTER}
    monkeypatch.setattr(remote_workers.automatic_connection, 'discover', attach)
    monkeypatch.setattr(remote_workers.home, 'device_id', lambda: 'device-a')
    helper_name = 'lan-launch-device-a'
    with pytest.raises(PermissionError):
        tokens.load(state.caller.base, helper_name)
    args = SimpleNamespace(project=str(state.roots['device-a']), device='device-b', agent='',
                           name='cli-qwen', model='auto', effort='off', max_effort='medium',
                           ctx='32k', max_output_tokens=2048, task='Report the shared project identity')
    result = remote_workers.run(args)
    assert result['identity'].startswith('lan-device-b-')
    assert result['requested_by'] == helper_name
    helper = RemoteWorkspace(state.caller.host, PROJECT, cluster=CLUSTER)
    helper_token = tokens.load(helper.base, helper_name)
    rows = helper.call('thread', helper_token, result['task_seq'])
    assert rows[0]['from'] == helper_name and rows[0]['to'] == result['identity']
    assert rows[0]['type'] == 'task' and 'shared project identity' in rows[0]['text']
    worker = localagent.load(state.local, 'cli-qwen')
    assert worker.orders_from == (helper_name,)
    helper_info = helper.call('whoami', helper_token)
    assert helper_info['model'] == '' and helper_info['harness'] == 'ml-stack-workspace'
    assert discovered == [state.roots['device-a']]
    assert len(state.spawned) == 1


@pytest.mark.redteam
def test_valid_capability_for_another_project_cannot_launch_a_worker(devices):
    workspace = devices.authority.workspace(PROJECT)
    entries = workspace.registry._load()
    entries['test-caller']['project']['key'] = 'b' * 32
    workspace.registry._save(entries)
    with pytest.raises(PermissionError):
        devices.target._request('worker', devices.body)
    assert not devices.spawned


@pytest.mark.redteam
def test_target_must_have_the_registered_project(devices):
    with pytest.raises(PermissionError):
        RemoteWorkspace(devices.target.host, 'b' * 32, cluster=CLUSTER)._request('worker', devices.body)
    assert not devices.spawned


def test_suffixed_worker_identity_is_reused_after_first_enrollment(devices):
    ws = devices.authority.workspace(PROJECT)
    scope = {'key': PROJECT, 'name': 'sample', 'cluster': CLUSTER, 'cluster_id': CLUSTER_ID}
    occupied, _ = ws.registry.enroll_project('worker-alias', scope, 3600)
    first = remote_workers.credential(devices.caller, occupied, 'Qwen3.8-test', 'device-a')
    actor = devices.caller.call('whoami', first)['id']
    assert actor != occupied
    second = remote_workers.credential(devices.caller, occupied, 'Qwen3.8-test', 'device-a')
    assert second == first
    assert devices.caller.call('whoami', second)['id'] == actor


@pytest.mark.redteam
def test_missing_recorded_worker_capability_cannot_reenroll(devices):
    first = remote_workers.credential(devices.caller, 'recorded-worker', 'Qwen3.8-test', 'device-a')
    actor = devices.caller.call('whoami', first)['id']
    (tokens.directory(devices.caller.base) / actor).unlink()
    before = devices.authority.workspace(PROJECT).registry.ids()
    with pytest.raises(PermissionError, match='unavailable'):
        remote_workers.credential(devices.caller, 'recorded-worker', 'Qwen3.8-test', 'device-a')
    assert devices.authority.workspace(PROJECT).registry.ids() == before


@pytest.mark.redteam
def test_worker_alias_database_cannot_follow_a_symlink(devices, tmp_path):
    destination = tmp_path / 'foreign.db'
    destination.write_text('preserve')
    (devices.caller.base / 'worker-identities.db').symlink_to(destination)
    with pytest.raises(PermissionError):
        remote_workers.credential(devices.caller, 'unsafe-worker', 'Qwen3.8-test', 'device-a')
    assert destination.read_text() == 'preserve'


@pytest.mark.redteam
def test_malformed_worker_alias_cannot_probe_paths_outside_session_storage(devices, monkeypatch):
    from ml_stack.graph.store import GraphStore
    path = devices.caller.base / 'worker-identities.db'
    with GraphStore(path) as graph:
        path.chmod(0o600)
        graph.upsert_node({'id': 'worker:malformed', 'kind': 'worker-identity', 'label': 'malformed',
                           'attrs': {'requested': 'malformed', 'identity': '../../foreign'}})
    with pytest.raises(PermissionError, match='identity record is invalid'):
        remote_workers.credential(devices.caller, 'malformed', 'Qwen3.8-test', 'device-a')


def test_generic_canonical_adapter_does_not_supply_a_worker_token_hook(devices):
    devices.target._request('worker', devices.body)
    agent = localagent.load(devices.local, 'local-qwen')
    adapter = remote_workers.CanonicalWorkspace(devices.caller, devices.token)
    adapter.base = devices.local.base
    loop = localloop.Loop(adapter, agent, localloop.Held(None, {'id': 'test-lease'}),
                         localloop.Settings(), (lambda: False, localagent.Status(devices.local, agent.name)))
    assert loop.token == tokens.load(devices.local.base, agent.identity)
    assert loop.token != devices.token


def _rotate_cluster(state, group=CLUSTER):
    key = base64.urlsafe_b64encode(bytes(reversed(range(32)))).rstrip(b'=')
    state.keyfile.write_bytes(key)
    _write_memberships([Membership(group, key, mode='dev')], state.keyfile)
    for peer in state.peers:
        peer.token = derive_token(key)
    return hashlib.sha256(key).hexdigest()


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('group', [CLUSTER, 'converged-development'])
def test_worker_alias_renews_after_dev_key_rotation_without_changing_identity(devices, legacy, group):
    from ml_stack.graph.store import GraphStore
    before = remote_workers.credential(devices.caller, 'rotating-worker', 'Qwen3.8-test', 'device-a')
    identity = devices.caller.call('whoami', before)['id']
    if legacy:
        with GraphStore(devices.caller.base / 'worker-identities.db') as graph:
            node = next(row for row in graph.nodes('worker-identity') if row['attrs']['identity'] == identity)
            attrs = dict(node['attrs'])
            attrs.pop('cluster_id')
            graph.upsert_node({**node, 'attrs': attrs})
    cluster_id = _rotate_cluster(devices, group)
    current = RemoteWorkspace(devices.caller.host, PROJECT, cluster=group)
    after = remote_workers.credential(current, 'rotating-worker', 'Qwen3.8-test', 'device-a')
    assert after == before
    who = current.call('whoami', after)
    assert who['id'] == identity and who['project']['cluster_id'] == cluster_id
    with GraphStore(current.base / 'worker-identities.db') as graph:
        assert next(row['attrs'] for row in graph.nodes('worker-identity')
                    if row['attrs']['identity'] == identity)['cluster_id'] == cluster_id


@pytest.mark.redteam
def test_rekey_cannot_restore_a_revoked_worker(devices):
    token = remote_workers.credential(devices.caller, 'revoked-worker', 'Qwen3.8-test', 'device-a')
    identity = devices.caller.call('whoami', token)['id']
    ws = devices.authority.workspace(PROJECT)
    records = ws.registry._load()
    records[identity]['revoked'] = True
    ws.registry._save(records)
    before = ws.registry.ids()
    _rotate_cluster(devices)
    current = RemoteWorkspace(devices.caller.host, PROJECT, cluster=CLUSTER)
    with pytest.raises(PermissionError):
        remote_workers.credential(current, 'revoked-worker', 'Qwen3.8-test', 'device-a')
    assert ws.registry.ids() == before


@pytest.mark.parametrize('group', [CLUSTER, 'converged-development'])
def test_live_board_worker_refreshes_tls_transport_and_sidecar_after_rekey(devices, group):
    result = devices.target._request('worker', devices.body)
    from ml_stack.files import read_json
    path = localagent.folder(devices.local) / 'local-qwen.remote.json'
    worker = remote_workers.BoardWorker(devices.local, read_json(path, {}))
    old_token = worker.token
    cluster_id = _rotate_cluster(devices, group)
    current = RemoteWorkspace(devices.caller.host, PROJECT, cluster=group)
    current.renew('test-caller')
    current.call('send', devices.token, result['identity'], 'question', 'Report after the cluster refresh')
    rows = worker.wait(old_token, 0, raw=True)
    assert any('cluster refresh' in row['raw'] for row in rows)
    worker.send(old_token, 'test-caller', 'answer', 'Same worker after the refresh', reply_to=rows[0]['seq'])
    assert any('Same worker' in row['raw'] for row in current.call('inbox', devices.token, raw=True))
    assert worker.auth(old_token).id == result['identity'] and worker.token == old_token
    assert read_json(path, {})['cluster_id'] == cluster_id
    assert len(devices.spawned) == 1


@pytest.mark.redteam
def test_explicit_missing_cli_identity_cannot_enroll_a_replacement(devices, monkeypatch):
    monkeypatch.setattr(remote_workers, 'selected', lambda root: {'host': devices.caller.host,
                        'project_id': PROJECT, 'cluster': CLUSTER})
    monkeypatch.setattr(remote_workers.home, 'device_id', lambda: 'device-a')
    args = SimpleNamespace(project=str(devices.roots['device-a']), device='device-b', agent='missing-caller',
                           name='missing-qwen', model='auto', effort='off', max_effort='medium',
                           ctx='32k', max_output_tokens=2048, task='')
    before = devices.authority.workspace(PROJECT).registry.ids()
    with pytest.raises(PermissionError):
        remote_workers.run(args)
    assert devices.authority.workspace(PROJECT).registry.ids() == before and not devices.spawned


def test_restarted_worker_renews_saved_scope_after_dev_group_convergence(devices):
    from ml_stack.files import read_json
    result = devices.target._request('worker', devices.body)
    path = localagent.folder(devices.local) / 'local-qwen.remote.json'
    record = read_json(path, {})
    cluster_id = _rotate_cluster(devices, 'converged-development')
    worker = remote_workers.BoardWorker(devices.local, record)
    assert worker.auth(worker.token).id == result['identity']
    assert read_json(path, {})['cluster'] == 'converged-development'
    assert read_json(path, {})['cluster_id'] == cluster_id
    assert len(devices.spawned) == 1


@pytest.mark.redteam
def test_running_worker_cannot_adopt_another_board_certificate_during_rekey(devices):
    from ml_stack.files import read_json
    devices.target._request('worker', devices.body)
    path = localagent.folder(devices.local) / 'local-qwen.remote.json'
    worker = remote_workers.BoardWorker(devices.local, read_json(path, {}))
    _rotate_cluster(devices)
    devices.peers[0].beacon.cert = devices.peers[1].beacon.cert
    with pytest.raises(PermissionError, match='cannot switch'):
        worker.auth(worker.token)


def test_self_lan_authority_uses_the_signed_loopback_certificate(devices, monkeypatch):
    state = devices
    monkeypatch.setattr(remote_workers.home, 'machine_id', lambda: 'device-a')
    authority_peer = state.peers[0]
    monkeypatch.setattr(authority_peer, 'base_url', f'https://127.0.0.1:{authority_peer.beacon.port}')
    lan_host = f'https://{discovery.primary_ip()}:{authority_peer.beacon.port}'
    state.project.board_host = lan_host
    caller = RemoteWorkspace(lan_host, PROJECT, cluster=CLUSTER)
    assert caller.device_cert == authority_peer.beacon.cert
    assert caller.call('whoami', state.token)['id'] == 'test-caller'
    result = caller._request('worker', state.body)
    assert result['identity'].startswith('lan-device-a-')
    assert len(state.spawned) == 1
    http.pin(urlsplit(lan_host).netloc, None)


@pytest.mark.redteam
@pytest.mark.parametrize('wrong', ['machine', 'certificate', 'port', 'address'])
def test_self_lan_alias_rejects_unbound_endpoints(devices, monkeypatch, wrong):
    state = devices
    monkeypatch.setattr(remote_workers.home, 'machine_id', lambda: 'device-a')
    node = state.peers[0]
    cert = node.beacon.cert if wrong != 'certificate' else ''
    beacon = SimpleNamespace(machine='other-node' if wrong == 'machine' else 'device-a', cert=cert)
    observed = SimpleNamespace(base_url=f'https://127.0.0.1:{node.beacon.port}', beacon=beacon)
    monkeypatch.setattr(Peer, 'discover', lambda **kwargs: [observed])
    host = '192.168.99.99' if wrong == 'address' else discovery.primary_ip()
    port = node.beacon.port + (1 if wrong == 'port' else 0)
    with pytest.raises(PermissionError, match='authenticated'):
        RemoteWorkspace(f'https://{host}:{port}', PROJECT, cluster=CLUSTER)


@pytest.mark.redteam
def test_cli_excludes_own_logical_node_with_a_distinct_physical_identity(devices, monkeypatch):
    state = devices
    monkeypatch.setattr(remote_workers.home, 'machine_id', lambda: 'device-a')
    monkeypatch.setattr(remote_workers.home, 'device_id', lambda: 'f' * 64)
    monkeypatch.setattr(remote_workers, 'selected', lambda root:
                        {'host': state.caller.host, 'project_id': PROJECT, 'cluster': CLUSTER})
    args = SimpleNamespace(project=str(state.roots['device-a']), device='device-a', agent='')
    with pytest.raises(PermissionError, match='select one discovered remote device'):
        remote_workers.run(args)
    assert not state.spawned


@pytest.mark.redteam
def test_self_lan_alias_transport_refuses_another_signed_certificate(devices, monkeypatch):
    state = devices
    monkeypatch.setattr(remote_workers.home, 'machine_id', lambda: 'device-a')
    node = state.peers[0]
    forged = SimpleNamespace(base_url=f'https://127.0.0.1:{node.beacon.port}',
                             beacon=SimpleNamespace(machine='device-a', cert=state.peers[1].beacon.cert))
    monkeypatch.setattr(Peer, 'discover', lambda **kwargs: [forged])
    host = f'https://{discovery.primary_ip()}:{node.beacon.port}'
    try:
        caller = RemoteWorkspace(host, PROJECT, cluster=CLUSTER)
        with pytest.raises(PermissionError, match='CERTIFICATE_VERIFY_FAILED'):
            caller.call('whoami', state.token)
        assert not state.spawned
    finally:
        http.pin(urlsplit(host).netloc, None)


@pytest.mark.redteam
@pytest.mark.parametrize('binding', ['absent', 'certificate', 'machine', 'userinfo', 'path'])
def test_exact_endpoint_requires_a_signed_device_origin(devices, binding):
    from ml_stack.fleet.remote import device_address
    node = devices.peers[0]
    beacon = SimpleNamespace(machine=node.beacon.machine, cert=node.beacon.cert)
    host = node.base_url
    if binding == 'absent':
        beacon = None
    elif binding == 'certificate':
        beacon.cert = ''
    elif binding == 'machine':
        beacon.machine = ''
    elif binding == 'userinfo':
        host = host.replace('https://', 'https://other@')
    else:
        host += '/unrelated'
    observed = SimpleNamespace(base_url=host, beacon=beacon)
    assert not device_address(observed, host)


@pytest.mark.parametrize('label', [
    'Qwen3.8-27B (Q4_K_XL)',
    'hf:' + 'downloaded-model-namespace-' * 4 + '/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf',
])
def test_tls_worker_normalizes_model_claim_and_preserves_full_runtime_reference(devices, monkeypatch, label):
    state = devices
    ref = 'hf:' + 'downloaded-model-namespace-' * 4 + '/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf'
    pick = localmodel.Pick(ref=ref, name=label, size_bytes=1)
    monkeypatch.setattr(localmodel, 'choose', lambda *args, **kwargs: pick)
    result = state.target._request('worker', {**state.body, 'model': ref})
    worker = localagent.load(state.local, 'local-qwen')
    assert worker.model == ref and worker.model_name == label
    info = state.authority.workspace(PROJECT).registry.info(result['identity'])
    assert info['model'] == localmodel.model_identity(label)
    local_info = state.local.registry.info(worker.identity)
    assert local_info['model'] == info['model']
    monkeypatch.setattr(localagent, 'alive', lambda agent: True)
    again = state.target._request('worker', {**state.body, 'model': ref})
    assert again['already'] and again['identity'] == result['identity']
    assert len(state.spawned) == 1


def test_remote_none_and_large_explicit_task_limits_preserve_shape():
    from ml_stack.workspace import remote_workers
    body = {"cluster": "dev", "cluster_id": "id", "max_output_tokens": None,
            "task_caps": {"seconds": 200000, "rounds": None}}
    assert remote_workers._settings(body, ("dev", "id")) == "local-qwen"
    for caps in ({"calls": True}, {"seconds": float("inf")}, {"unknown": None}):
        with pytest.raises(ValueError):
            remote_workers._settings({**body, "task_caps": caps}, ("dev", "id"))


def test_connection_acl_failure_preserves_destination_without_credential_bytes(tmp_path, monkeypatch):
    path = tmp_path / "connection.json"
    remote_workers._save_connection(path, {"identity": "existing"})
    before = path.read_bytes()
    seen = []
    def refuse(temporary):
        seen.append(temporary.read_bytes())
        raise OSError("temporary ACL denied")
    if remote_workers.os.name != "nt":
        pytest.skip("Windows ACL behavior")
    monkeypatch.setattr(remote_workers, "restrict", refuse)
    with pytest.raises(OSError, match="temporary ACL denied"):
        remote_workers._save_connection(path, {"identity": "replacement"})
    assert seen == [b""]
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_cli_prints_authenticated_followup_and_effective_limits(monkeypatch, capsys, tmp_path):
    reply = {'identity': 'worker', 'state': 'running', 'model': 'Qwen3.8-test',
             'requested_by': 'caller', 'task_seq': 17, 'requested_context': 'auto',
             'effective_limits': {'max_output_tokens': None, 'task_caps': {'rounds': None}}}
    monkeypatch.setattr(remote_workers, 'run', lambda args: reply)
    project = tmp_path / 'project space'
    assert remote_workers.main_cli(SimpleNamespace(json=False, device='target', project=str(project))) == 0
    output = capsys.readouterr().out
    assert 'Caller: caller' in output
    assert 'Requested context: auto' in output
    assert f"cd '{project.resolve()}' && ml-stack-workspace thread 17 --agent caller" in output
    assert '"max_output_tokens": null' in output
