"""Two device roots use authenticated Fleet transport and one authoritative graph."""

import base64
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest
import wheel_cache
from workspace_kit import Kit, clean_env

from ml_stack import http
from ml_stack.fleet import projects, tls
from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.discovery import derive_token, mint_cluster
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.onboard.requests import Device, Devices
from ml_stack.fleet.remote import Peer
from ml_stack.http import Server, ServerError
from ml_stack.hub.peerbook import PeerBook
from ml_stack.net import git
from ml_stack.workspace import cli, coordinator, coordinator_client, coordinator_config, tokens
from ml_stack.workspace.coordination import workspace_id
from ml_stack.workspace.coordinator_client import Remote
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.taskboard import TaskBoard


@pytest.fixture(scope='module')
def installed_metadata():
    return wheel_cache.built() / 'installed'


@pytest.fixture
def shared(tmp_path, monkeypatch, installed_metadata):
    monkeypatch.syspath_prepend(str(installed_metadata))
    kit = Kit(clean_env(monkeypatch, tmp_path / 'coordinator'))
    kit.limits(sends_per_window=1000)
    kit.alice, kit.bob = kit.agent('alice'), kit.agent('bob')
    identity = workspace_id(kit.ws)
    coordinator_config.save(kit.base, {'mode': 'host', 'workspace': identity})
    runner = JobRunner(tmp_path / 'daemon')
    membership = mint_cluster('default', tmp_path / 'cluster.key')
    monkeypatch.setenv('ML_STACK_CLUSTER_KEY', str(tmp_path / 'cluster.key'))
    fleet_token = derive_token(membership.key)
    kit.project_dir = tmp_path / 'shared-project'
    kit.project_dir.mkdir()
    git.run(['init', str(kit.project_dir)])
    git.run(['remote', 'add', 'origin', 'https://example.invalid/shared-project.git'], cwd=kit.project_dir)
    kit.project_scope = {'key': projects.identity(kit.project_dir), 'name': 'shared-project'}
    hosted = projects.ProjectRegistry(tmp_path / 'projects', 'coordinator-device', host='https://coordinator:8770')
    hosted._projects[kit.project_scope['key']] = projects.Project(
        id=kit.project_scope['key'], name=kit.project_scope['name'], root=str(kit.project_dir),
        source_machine=hosted.machine, board_host=hosted.host, shared=True)
    hosted._save()
    device = Device('d' * 64, 'paired-device', 'paired-host', '127.0.0.1', 1,
                    mine=True, secret=base64.urlsafe_b64encode(b'd' * 32).decode())
    server = Server(('127.0.0.1', 0), make_handler(Daemon(
        runner, tmp_path / 'files', fleet_token, devices=lambda: [device], projects=hosted)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    peer = Peer(f'http://127.0.0.1:{server.server_address[1]}', fleet_token)
    kit.remote = Remote({'workspace': identity, 'endpoint': peer.base_url}, peer)
    kit.device = tmp_path / 'windows-device'
    kit.device.mkdir()
    installed_home = tmp_path / 'installed-home'
    monkeypatch.setenv('ML_STACK_HOME', str(installed_home))
    certificate = tls.identity(tmp_path / 'tls', 'test-coordinator').beacon
    Devices(installed_home / 'onboard' / 'devices.json')._write([device])
    PeerBook(installed_home / 'onboard' / 'peers.json').add({
        'name': 'test-coordinator', 'url': peer.base_url, 'source': 'pairing',
        'fingerprint': device.fingerprint, 'certificate': certificate,
        'device_secret': device.secret})
    coordinator_config.save(kit.device, {'mode': 'remote', 'workspace': identity,
                                         'endpoint': peer.base_url, 'cert': certificate})
    tokens.store(kit.device, 'alice', kit.alice)
    try:
        yield kit
    finally:
        server.shutdown()
        server.server_close()
        runner.shutdown()


def test_real_distribution_registration_and_two_roots_share_messages_tasks_claims(shared):
    from importlib.metadata import entry_points
    assert any(entry.value == 'ml_stack.workspace.coordinator:route'
               for entry in entry_points(group='ml_stack.peer_routes'))
    sent = shared.remote.command(['send', 'bob', 'task', 'Inspect shared state'], shared.alice)
    assert shared.ws.inbox(shared.bob)[0]['seq'] == sent['seq']
    assert shared.remote.command(['inbox'], shared.bob)[0]['seq'] == sent['seq']
    claim = shared.remote.command(['claim', 'branch', 'development-work'], shared.alice)
    assert shared.ws.who_owns('branch', 'development-work')['owner'] == claim['owner'] == 'alice'
    assert shared.remote.command(['who', 'branch', 'development-work'], shared.bob)['owner'] == 'alice'
    spec = {'title': 'Shared task', 'description': 'Read coordinator graph', 'acceptance': ['Exact graph state'],
            'project': {}, 'capabilities': [], 'limits': {}, 'source_key': 'two-device-proof'}
    task = shared.remote.command(['task-create', json.dumps(spec)], shared.alice)
    assert TaskBoard(shared.ws).get(shared.alice, task['id'])['id'] == task['id']
    assert shared.remote.command(['tasks'], shared.alice)['tasks'][0]['id'] == task['id']
    assert not (shared.device / 'messages.jsonl').exists()
    assert not (shared.device / 'coordination.db').exists()


def test_exact_request_race_and_reconnect_append_once(shared):
    request = uuid4().hex
    def send(_):
        return shared.remote.command(['send', 'bob', 'task', 'One assigned task'], shared.alice, request_id=request)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(send, range(2)))
    assert results[0]['seq'] == results[1]['seq']
    assert len(shared.ws.bus.outbox('alice')) == 1
    assert send(None)['seq'] == results[0]['seq']
    with pytest.raises(ServerError) as refused:
        shared.remote.command(['send', 'bob', 'task', 'Changed payload'], shared.alice, request_id=request)
    assert refused.value.status == 403


@pytest.mark.redteam
def test_live_agent_and_fleet_auth_and_recipient_rights_precede_replay(shared):
    request = uuid4().hex
    shared.remote.command(['send', 'bob', 'task', 'Original task'], shared.alice, request_id=request)
    shared.ws.revoke(shared.owner, 'bob')
    with pytest.raises(ServerError) as receiver:
        shared.remote.command(['send', 'bob', 'task', 'Original task'], shared.alice, request_id=request)
    assert receiver.value.status == 400
    shared.ws.revoke(shared.owner, 'alice')
    with pytest.raises(ServerError) as sender:
        shared.remote.command(['status'], shared.alice)
    assert sender.value.status == 403
    with pytest.raises(ServerError) as fleet:
        Remote(shared.remote.config, Peer(shared.remote.peer.base_url, 'wrong')).command(['status'], shared.bob)
    assert fleet.value.status == 401


@pytest.mark.redteam
@pytest.mark.parametrize('argv', [['mint', 'untrusted'], ['init'], ['agent', 'start'],
                                  ['task-integrate', 'task:' + 'a' * 32],
                                  ['whoami', '--agent', 'owner'], ['whoami', '--token-file', '/secret'],
                                  ['notes-verify'], ['claim', 'port', '8784']])
def test_remote_call_cannot_open_local_execution_or_choose_an_actor(shared, argv):
    with pytest.raises(ServerError) as refused:
        shared.remote.command(argv, shared.alice)
    assert refused.value.status == 403
    assert not shared.ws.registry.role_of('untrusted')


@pytest.mark.redteam
def test_person_token_wrong_workspace_bounds_and_plaintext_remote_are_refused(shared):
    with pytest.raises(ServerError) as person:
        shared.remote.command(['status'], shared.owner)
    assert person.value.status == 403
    with pytest.raises(ServerError) as wrong:
        Remote({**shared.remote.config, 'workspace': 'workspace:' + '0' * 32}, shared.remote.peer).command(['status'], shared.alice)
    assert wrong.value.status == 403
    with pytest.raises(ServerError) as large:
        shared.remote.command(['send', 'bob', 'status', 'x' * coordinator.MAX_REQUEST], shared.alice)
    assert large.value.status == 413
    from ml_stack.fleet.onboard.web import Call
    request = Call('GET', '/workspace/v1/info', {}, '192.0.2.1', False, lambda _max: b'')
    assert coordinator.answer(shared.ws, request)[0] == 403
    with pytest.raises(Denied):
        coordinator_config.validate_endpoint('http://192.0.2.1:8770')


def test_existing_invitation_enrolls_remote_agent_without_person_credentials(shared):
    invite = shared.ws.invites.create('windows', 60, {})
    name = shared.remote.join(shared.device, invite, 'windows', 'model-a', 'codex')
    token = tokens.load(shared.device, name)
    assert shared.remote.command(['whoami'], token)['id'] == name
    assert shared.ws.auth(token).role == 'agent'
    assert not (shared.device / 'agents.json').exists()
    with pytest.raises(ServerError):
        shared.remote.join(shared.device, invite, 'duplicate')


def test_declared_cli_arguments_roundtrip_without_credentials(shared):
    args = cli.COMMANDS.parser().parse_args(['send', 'bob', 'status', 'Progress', '--agent', 'alice',
                                             '--request-id', 'a' * 32])
    from ml_stack.workspace.coordinator_client import argv_for
    options = next(options for name, _help, options, _fn in cli.TABLE if name == args.cmd)
    argv = argv_for(args, [*cli.COMMON, *options])
    assert '--agent' not in argv and '--token-file' not in argv and '--request-id' not in argv
    got = shared.remote.command(argv, shared.alice, request_id=args.request_id)
    assert got['from'] == 'alice' and got['from_name'] == got['from_name'].strip()


def test_installed_cli_on_second_device_reads_shared_state_without_local_fallback(shared, installed_metadata):
    remote = coordinator_client.client(shared.device)
    agent = remote.ensure(shared.device, 'second-device', project=shared.project_scope)
    sent = shared.remote.command(['send', agent, 'task', 'Shared CLI task'], shared.bob)
    environment = {**os.environ, 'ML_STACK_WORKSPACE_HOME': str(shared.device),
                   'PYTHONPATH': str(installed_metadata), 'ML_STACK_WORKSPACE_AGENT': agent}
    result = subprocess.run([sys.executable, '-m', 'ml_stack.workspace.cli', 'inbox', '--json'],
                            cwd=shared.project_dir, env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)[0]['seq'] == sent['seq']
    registration = shared.ws.registry._load()[agent]
    assert registration['session_device'] == 'd' * 64
    assert registration['project'] == shared.project_scope
    denied = subprocess.run([sys.executable, '-m', 'ml_stack.workspace.cli', 'init', '--json'],
                            cwd=shared.project_dir, env=environment, capture_output=True, text=True, timeout=15)
    assert denied.returncode == 3
    assert not (shared.device / 'agents.json').exists()


@pytest.mark.parametrize('tool', ['workspace_status', 'workspace_inbox', 'workspace_tasks'])
def test_remote_mcp_never_falls_back_to_local_workspace(tmp_path, monkeypatch, tool):
    from ml_stack.workspace import tools

    base = clean_env(monkeypatch, tmp_path)
    coordinator_config.save(base, {'mode': 'remote', 'workspace': 'workspace:' + 'a' * 32,
                                     'endpoint': 'https://coordinator.example:8770'})
    with pytest.raises(Denied, match='not remote-enabled'):
        getattr(tools, tool)()
    assert not (base / 'registry.json').exists()


@pytest.mark.parametrize('handler', ['_nudging', '_hook', '_watching'])
def test_remote_watchers_refuse_before_opening_a_local_board(tmp_path, monkeypatch, handler):
    base = clean_env(monkeypatch, tmp_path)
    coordinator_config.save(base, {'mode': 'remote', 'workspace': 'workspace:' + 'a' * 32,
                                     'endpoint': 'https://coordinator.example:8770'})
    with pytest.raises(Denied, match='local-only'):
        getattr(cli, handler)(None)
    assert not (base / 'registry.json').exists()


@pytest.mark.redteam
@pytest.mark.parametrize('endpoint', [
    'http://192.0.2.10:8786', 'file:///etc/passwd', 'https://user:secret@example.org',
    'https://example.org/callback', 'https://example.org?token=secret',
    'https://example.org#authority', '//example.org', 'ftp://example.org'])
def test_hostile_saved_coordinator_origin_is_refused_before_transport(tmp_path, monkeypatch, endpoint):
    from ml_stack.workspace import coordinator_client

    (tmp_path / 'coordinator.json').write_text(json.dumps({
        'version': 1, 'mode': 'remote', 'workspace': 'workspace:' + 'a' * 32, 'endpoint': endpoint}))
    monkeypatch.setattr(coordinator_client, '_device_peer',
                        lambda config: pytest.fail('hostile configuration reached credential/transport setup'))
    with pytest.raises(Denied):
        coordinator_client.client(tmp_path)


def test_origin_attack_regression_detects_removed_validation_guard(tmp_path, monkeypatch):
    with monkeypatch.context() as changed:
        changed.setattr(coordinator_config, 'validate_endpoint', lambda _: None)
        with pytest.raises(pytest.fail.Exception, match='reached credential/transport setup'):
            test_hostile_saved_coordinator_origin_is_refused_before_transport(
                tmp_path, changed, 'http://192.0.2.10:8786')


def test_fleet_membership_without_coordinator_selection_preserves_local_workspace(tmp_path, monkeypatch):
    registry = tmp_path / 'agents.json'
    registry.write_text('{"existing": true}', encoding='utf-8')
    before = registry.read_bytes()
    monkeypatch.setattr(coordinator_client, 'load_cluster_key', lambda: bytes(range(32)))
    monkeypatch.setattr(coordinator_client, 'discover', lambda: [])
    monkeypatch.setattr(coordinator_client, '_device_peer',
                        lambda config: pytest.fail('unselected coordinator reached device proof'))
    assert coordinator_client.client(tmp_path) is None
    assert registry.read_bytes() == before
    assert not (tmp_path / 'coordinator.json').exists()


def test_selected_coordinator_outage_preserves_remote_authority(tmp_path, monkeypatch):
    coordinator_config.save(tmp_path, {'mode': 'remote', 'workspace': 'workspace:' + 'a' * 32,
                                      'endpoint': 'https://coordinator.example:8770'})
    before = (tmp_path / 'coordinator.json').read_bytes()
    monkeypatch.setattr(coordinator_client, 'discover', lambda: [])

    def unavailable(config):
        raise Denied('selected device is unavailable')

    monkeypatch.setattr(coordinator_client, '_device_peer', unavailable)
    with pytest.raises(Denied, match='selected coordinator'):
        coordinator_client.client(tmp_path)
    assert (tmp_path / 'coordinator.json').read_bytes() == before
    assert not (tmp_path / 'agents.json').exists()


def test_coordinator_discovery_without_pairing_does_not_probe_network(monkeypatch):
    monkeypatch.setattr(coordinator_client, '_paired_rows', lambda: [])
    monkeypatch.setattr(Peer, 'discover',
                        classmethod(lambda cls, **kwargs: pytest.fail('unpaired discovery')))
    assert coordinator_client.discover() == []


def test_coordinator_discovery_rejects_unpaired_peer_before_http(monkeypatch):
    peer = SimpleNamespace(base_url='https://127.0.0.1:8770',
                           beacon=SimpleNamespace(cert='invented-cert'))
    monkeypatch.setattr(coordinator_client, '_paired_rows', lambda: [{}])
    monkeypatch.setattr(Peer, 'discover', classmethod(lambda cls, **kwargs: [peer]))

    def denied(config):
        raise Denied('no active pairing')

    monkeypatch.setattr(coordinator_client, '_device_peer', denied)
    assert coordinator_client.discover() == []


@pytest.mark.parametrize('response', ['valid', 'timeout', 'list'])
def test_coordinator_discovery_uses_bounded_device_transport(monkeypatch, response):
    peer = SimpleNamespace(base_url='https://127.0.0.1:8770',
                           beacon=SimpleNamespace(cert='invented-cert'))
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if response == 'timeout':
            raise ServerError('timeout', status=503)
        info = {'authority': 'coordinator', 'protocol': 1, 'workspace': 'workspace:' + 'a' * 32}
        return 200, json.dumps([] if response == 'list' else info).encode(), {}

    monkeypatch.setattr(coordinator_client, '_paired_rows', lambda: [{}])
    monkeypatch.setattr(Peer, 'discover', classmethod(lambda cls, **kwargs: [peer]))
    monkeypatch.setattr(coordinator_client, '_device_peer',
                        lambda config: SimpleNamespace(_request=request))
    assert bool(coordinator_client.discover()) is (response == 'valid')
    assert calls == [('GET', '/workspace/v1/info', {'timeout': 2, 'retry': http.ONCE})]


def test_missing_selection_for_existing_remote_session_refuses_local_authority(tmp_path, monkeypatch):
    marker = tmp_path / 'remote-sessions.db'
    marker.write_bytes(b'existing remote session')
    monkeypatch.setattr(coordinator_client, 'discover',
                        lambda: pytest.fail('missing selection reached discovery'))
    with pytest.raises(Denied, match='selection is missing'):
        coordinator_client.client(tmp_path)
    assert marker.read_bytes() == b'existing remote session'
    assert not (tmp_path / 'agents.json').exists()
@pytest.mark.parametrize('path', [
    '/workspace/v1/projects', '/workspace/v1/projects/' + 'a' * 32 + '/snapshot?hash=revision',
    '/workspace/v1/projects/' + 'a' * 32 + '/join',
    '/workspace/v1/projects/' + 'a' * 32 + '/board',
    '/workspace/v1/local-project', '/workspace/v1/unknown', '/workspace/v1/info/extra',
])
def test_non_coordinator_routes_fall_through_without_opening_workspace(monkeypatch, path):
    def forbidden():
        pytest.fail('an unrelated route must not open the coordinator workspace')
    monkeypatch.setattr(coordinator, 'Workspace', forbidden)
    assert coordinator.route(SimpleNamespace(path=path), b'project request') is False
