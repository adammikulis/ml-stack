"""Two device roots use authenticated Fleet transport and one authoritative graph."""

import concurrent.futures
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from uuid import uuid4

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.discovery import derive_token, mint_cluster
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.remote import Peer
from ml_stack.http import Server, ServerError
from ml_stack.workspace import cli, coordinator, coordinator_config, tokens
from ml_stack.workspace.coordination import workspace_id
from ml_stack.workspace.coordinator_client import Remote
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.taskboard import TaskBoard


@pytest.fixture(scope='module')
def installed_metadata(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp('coordinator-wheel')
    environment = {**os.environ, 'PIP_NO_INDEX': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1'}
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                    '--no-cache-dir', '--wheel-dir', str(output), str(root)],
                   env=environment, check=True, capture_output=True, text=True, timeout=60)
    target = output / 'installed'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index',
                    '--target', str(target), str(next(output.glob('*.whl')))],
                   env=environment, check=True, capture_output=True, text=True, timeout=60)
    return target


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
    server = Server(('127.0.0.1', 0), make_handler(Daemon(runner, tmp_path / 'files', fleet_token)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    peer = Peer(f'http://127.0.0.1:{server.server_address[1]}', fleet_token)
    kit.remote = Remote({'workspace': identity, 'endpoint': peer.base_url}, peer)
    kit.device = tmp_path / 'windows-device'
    kit.device.mkdir()
    coordinator_config.save(kit.device, {'mode': 'remote', 'workspace': identity, 'endpoint': peer.base_url})
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
                                             '--label', 'helper', '--request-id', 'a' * 32])
    from ml_stack.workspace.coordinator_client import argv_for
    options = next(options for name, _help, options, _fn in cli.TABLE if name == args.cmd)
    argv = argv_for(args, [*cli.COMMON, *options])
    assert '--agent' not in argv and '--token-file' not in argv and '--request-id' not in argv
    got = shared.remote.command(argv, shared.alice, request_id=args.request_id)
    assert got['from'] == 'alice' and got['from_label'] == 'alice/helper'


def test_installed_cli_on_second_device_reads_shared_state_without_local_fallback(shared, installed_metadata):
    sent = shared.remote.command(['send', 'alice', 'task', 'Shared CLI task'], shared.bob)
    environment = {**os.environ, 'ML_STACK_WORKSPACE_HOME': str(shared.device),
                   'PYTHONPATH': str(installed_metadata), 'ML_STACK_WORKSPACE_AGENT': 'alice'}
    result = subprocess.run([sys.executable, '-m', 'ml_stack.workspace.cli', 'inbox', '--json'],
                            env=environment, capture_output=True, text=True, timeout=15, check=True)
    assert json.loads(result.stdout)[0]['seq'] == sent['seq']
    denied = subprocess.run([sys.executable, '-m', 'ml_stack.workspace.cli', 'init', '--json'],
                            env=environment, capture_output=True, text=True, timeout=15)
    assert denied.returncode == 3
    assert not (shared.device / 'agents.json').exists()
