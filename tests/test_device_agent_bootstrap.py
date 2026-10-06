"""Paired-device agent recovery preserves scope and denies revoked trust."""

import base64
import json
import threading
from types import SimpleNamespace

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.fleet import device_auth
from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.remote import Peer
from ml_stack.fleet.onboard.requests import Device
from ml_stack.fleet.onboard.web import Call
from ml_stack.workspace import coordinator, coordinator_config, device_sessions, tokens
from ml_stack.workspace.coordination import workspace_id
from ml_stack.workspace.identity import AGENT, Denied
from ml_stack.http import Server, ServerError
from ml_stack.workspace.coordinator_client import Remote
from ml_stack.workspace.remote_host import WorkspaceHost


@pytest.fixture
def enrolled(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    device = Device('a' * 64, 'test-device', 'test-host', '127.0.0.1', 1,
                    mine=True, secret=base64.urlsafe_b64encode(b'x' * 32).decode())
    projects = SimpleNamespace(machine='authority', list=lambda: [
        {'id': 'b' * 32, 'name': 'example', 'authority_machine': 'authority'}])
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
        device_sessions.check(kit.ws, token, device)
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
        device_sessions.check(kit.ws, second, device)


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


def test_canonical_project_bootstrap_has_same_device_boundary(enrolled):
    kit, device, projects, document = enrolled
    projects.workspace_base = lambda project_id: kit.base
    host = WorkspaceHost(projects)
    body = {**document, 'agent_token': ''}
    project_id = document['project']['key']
    assert host.answer(project_id, 'ensure', body)[0] == 403
    status, result = host.answer(project_id, 'ensure', body, device=device)
    assert status == 200
    request = {'agent_token': result['token'], 'operation': 'whoami', 'args': [], 'kwargs': {}}
    assert host.answer(project_id, 'board', request, device=device)[0] == 200
    assert host.answer(project_id, 'board', request)[0] == 403
