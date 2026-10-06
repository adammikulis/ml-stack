"""Installed device trust selects existing workspace authority without person credentials."""

import base64
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from workspace_kit import Kit

from ml_stack import home
from ml_stack.fleet import daemon, tls
from ml_stack.fleet.onboard import cli as onboard
from ml_stack.fleet.onboard.requests import Device, Devices
from ml_stack.fleet.remote import Peer
from ml_stack.hub.peerbook import PeerBook
from ml_stack.workspace import (
    coordinator_client,
    coordinator_config,
    coordinator_routes,
    device_agent,
    tokens,
)
from ml_stack.workspace.identity import AGENT, Denied


@pytest.fixture(autouse=True)
def installed_device(monkeypatch):
    monkeypatch.setattr(device_agent, 'device_id', lambda: '1234567890abcdef')


def own(kit):
    token = kit.ws.registry._add('local-account', 'launcher', AGENT, 0)
    tokens.store(kit.base, 'launcher', token)
    return token


def test_own_local_session_hosts_without_reading_person_token(tmp_path, monkeypatch):
    kit = Kit(tmp_path / 'workspace')
    token = own(kit)
    monkeypatch.setattr(coordinator_client, '_client', lambda base: None)
    read = tokens.read_file
    def agent_only(path):
        assert path.name != tokens.OWNER_FILE
        return read(path)
    monkeypatch.setattr(tokens, 'read_file', agent_only)
    result = coordinator_routes.ensure_host(kit.ws, token)
    assert result['mode'] == 'host'
    assert coordinator_routes.ensure_host(kit.ws, token) == result


@pytest.mark.redteam
@pytest.mark.parametrize('kind', ['foreign', 'network', 'child', 'revoked'])
def test_only_protected_own_local_identity_selects_authority(tmp_path, monkeypatch, kind):
    kit = Kit(tmp_path / 'workspace')
    token = own(kit)
    if kind == 'foreign':
        token = kit.agent('other')
        tokens.store(kit.base, 'other', token)
    elif kind == 'network':
        rows = kit.ws.registry._load()
        rows['launcher']['session_device'] = 'a' * 64
        kit.ws.registry._save(rows)
    elif kind == 'child':
        child = kit.ws.delegate(token, 'worker')
        token = tokens.load(kit.base, child['id'])
    else:
        kit.ws.revoke(kit.owner, 'launcher')
    with pytest.raises(Denied):
        coordinator_routes.change(kit.ws, token, {'action': 'host'})
    assert coordinator_config.load(kit.base) == {}


@pytest.mark.redteam
def test_remote_and_unavailable_shared_authority_never_becomes_local_host(tmp_path, monkeypatch):
    kit = Kit(tmp_path / 'workspace')
    token = own(kit)
    config = coordinator_config.save(kit.base, {'mode': 'remote', 'workspace': 'workspace:foreign',
                                               'endpoint': 'https://example.invalid:8784'})
    with pytest.raises(Denied):
        coordinator_routes.ensure_host(kit.ws, token)
    assert coordinator_config.load(kit.base) == config
    (kit.base / 'coordinator.json').unlink()
    def unavailable(base):
        raise Denied('the enrolled workspace coordinator is unavailable')
    monkeypatch.setattr(coordinator_client, '_client', unavailable)
    with pytest.raises(Denied, match='unavailable'):
        coordinator_routes.ensure_host(kit.ws, token)
    assert coordinator_config.load(kit.base) == {}


def test_own_agent_connect_preserves_existing_authority(tmp_path, monkeypatch):
    kit = Kit(tmp_path / 'workspace')
    token = own(kit)
    config = coordinator_config.save(kit.base, {'mode': 'host', 'workspace': 'workspace:existing'})
    monkeypatch.setattr(coordinator_client, 'connect', lambda *args: pytest.fail('authority replacement'))
    with pytest.raises(Denied, match='cannot be replaced'):
        coordinator_routes.change(kit.ws, token, {'action': 'connect', 'name': 'another'})
    assert coordinator_config.load(kit.base) == config


@pytest.mark.parametrize('direct', [False, True])
def test_concurrent_host_and_connect_cannot_replace_authority(tmp_path, monkeypatch, direct):
    kit = Kit(tmp_path / 'workspace')
    token = own(kit)
    saving, release, attempting = threading.Event(), threading.Event(), threading.Event()
    held, save = coordinator_routes.held, coordinator_config.save
    errors = []
    @contextmanager
    def observed_lock(path):
        if threading.current_thread().name == 'connecting':
            attempting.set()
        with held(path):
            yield
    def paused_save(base, document):
        saving.set()
        assert release.wait(5)
        return save(base, document)
    def select(document):
        try:
            if direct and document['action'] == 'connect':
                coordinator_client.connect(kit.base, document['name'])
            else:
                coordinator_routes.change(kit.ws, token, document)
        except Denied as error:
            errors.append(error)
    monkeypatch.setattr(coordinator_routes, 'held', observed_lock)
    monkeypatch.setattr(coordinator_client, 'held', observed_lock)
    monkeypatch.setattr(coordinator_config, 'save', paused_save)
    advertised = Peer('https://127.0.0.1:8786', 'cluster-proof',
                      beacon=SimpleNamespace(name='coordinator', cert='certificate'))
    monkeypatch.setattr(coordinator_client, 'discover', lambda: [(advertised, {'workspace': 'workspace:remote'})])
    monkeypatch.setattr(coordinator_client, '_device_peer', lambda config: pytest.fail('overwrote host'))
    host = threading.Thread(target=select, args=({'action': 'host'},))
    connect = threading.Thread(name='connecting', target=select,
                               args=({'action': 'connect', 'name': 'coordinator'},))
    host.start()
    try:
        assert saving.wait(5)
        connect.start()
        assert attempting.wait(5)
    finally:
        release.set()
        host.join(5)
        if connect.ident:
            connect.join(5)
    assert not host.is_alive() and not connect.is_alive()
    assert len(errors) == 1
    assert coordinator_config.load(kit.base)['mode'] == 'host'


def test_production_daemon_reuses_pairing_certificate_and_isolated_roots_stay_separate(tmp_path, monkeypatch):
    monkeypatch.setenv('ML_STACK_HOME', str(tmp_path / 'installation'))
    paired = onboard._identity(home.state('onboard'))
    serving = tls.identity(daemon.identity_directory(daemon.default_root()), 'daemon')
    assert serving.beacon == paired.beacon
    isolated = tmp_path / 'isolated-daemon'
    assert daemon.identity_directory(isolated) == isolated / 'tls'


@pytest.mark.parametrize('mismatch', ['', 'workspace', 'endpoint', 'name', 'revoked', 'certificate', 'offline'])
def test_saved_daemon_certificate_recovers_only_same_active_paired_authority(tmp_path, monkeypatch, mismatch):
    monkeypatch.setenv('ML_STACK_HOME', str(tmp_path / 'installation'))
    base = tmp_path / 'workspace'
    base.mkdir()
    paired = onboard._identity(home.state('onboard'))
    old = tls.identity(tmp_path / 'old-daemon', 'daemon')
    device = Device('d' * 64, 'coordinator', 'host', '127.0.0.1', 1,
                    mine=True, secret=base64.urlsafe_b64encode(b'd' * 32).decode(),
                    status='revoked' if mismatch == 'revoked' else 'active')
    Devices(home.state('onboard', 'devices.json'))._write([device])
    PeerBook(home.state('onboard', 'peers.json')).add({
        'name': 'coordinator', 'url': 'https://127.0.0.1:8786', 'source': 'pairing',
        'fingerprint': device.fingerprint, 'certificate': paired.beacon,
        'device_secret': device.secret})
    config = coordinator_config.save(base, {'mode': 'remote', 'workspace': 'workspace:existing',
                                            'name': 'coordinator', 'endpoint': 'https://127.0.0.1:8786',
                                            'cert': old.beacon})
    beacon = SimpleNamespace(name='other' if mismatch == 'name' else 'coordinator',
                             cert=old.beacon if mismatch == 'certificate' else paired.beacon)
    advertised = Peer('https://127.0.0.1:8787' if mismatch == 'endpoint' else config['endpoint'],
                      'cluster-proof', beacon=beacon)
    found = [(advertised, {
        'workspace': 'workspace:other' if mismatch == 'workspace' else config['workspace']})]
    monkeypatch.setattr(coordinator_client, 'discover', lambda: [] if mismatch == 'offline' else found)
    if mismatch:
        with pytest.raises(Denied):
            coordinator_client.client(base)
        assert coordinator_config.load(base) == config
    else:
        remote = coordinator_client.client(base)
        assert remote.config['cert'] == paired.beacon
        assert remote.config['workspace'] == config['workspace']
        assert remote.config['endpoint'] == config['endpoint']
