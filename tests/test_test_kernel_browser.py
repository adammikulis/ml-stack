"""Browser asset sealing and supervisor listener ownership."""
from __future__ import annotations

import contextlib
import os
import socket
import threading

import pytest
from test_fixture_plan import resources
from test_kernel_browser import Assets, identity, prepare, seal
from test_kernel_browser_client import Listener
from test_kernel_browser_endpoints import EndpointBank, local_interface


def test_browser_assets_are_sealed_and_source_changes_do_not_change_sealed_bytes(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    executable = source / 'browser'
    executable.write_bytes(b'owned browser fixture')
    executable.chmod(0o700)
    files, digest = seal(source, tmp_path / 'sealed', ())
    assert len(digest) == 64 and len(files) == 1
    assert files[0].read_bytes() == b'owned browser fixture'
    assert files[0].stat().st_mode & 0o777 == 0o500
    assets = Assets(files=files, identities={path: identity(path) for path in files})
    executable.write_bytes(b'changed source')
    assets.recheck()
    files[0].chmod(0o600)
    with pytest.raises(RuntimeError, match='identity changed'):
        assets.recheck()


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'writable'])
def test_browser_asset_seal_refuses_redirects_aliases_and_foreign_writers(tmp_path, kind):
    source = tmp_path / 'source'
    source.mkdir()
    file = source / 'browser'
    file.write_bytes(b'fixture')
    if kind == 'symlink':
        (source / 'alias').symlink_to(file)
    elif kind == 'hardlink':
        os.link(file, source / 'alias')
    else:
        file.chmod(0o666)
    with pytest.raises(RuntimeError, match=r'linked asset|multiply linked|writable'):
        seal(source, tmp_path / 'sealed', ())


def test_browser_resource_bundle_requires_supervisor_fixture_admission(tmp_path):
    assert prepare(frozenset(), tmp_path, {}, (), ()) is None


@pytest.fixture
@resources('browser-endpoints')
def bank():
    for key in ('DEV_TEST_BROWSER_ENDPOINT', 'DEV_TEST_BROWSER_IDENTITY',
                'DEV_TEST_BROWSER_TOKEN', 'DEV_TEST_REMOTE_LEASE'):
        assert os.environ.get(key), f'missing maintained browser fixture admission: {key}'


def test_transferred_listener_accepts_only_its_reserved_endpoint_and_releases(bank):
    listener = Listener()
    try:
        listener.socket.settimeout(2)
        assert listener.socket.getsockname()[0] == '127.0.0.1'
        assert listener.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
        with socket.create_connection(listener.socket.getsockname(), timeout=2) as client:
            peer, _ = listener.socket.accept()
            with peer:
                client.sendall(b'fixture')
                assert peer.recv(7) == b'fixture'
                peer.sendall(b'received')
                assert client.recv(8) == b'received'
    finally:
        listener.close()
    assert listener.socket is None


def test_endpoint_bank_refuses_invalid_requests_before_resource_access():
    bank = EndpointBank.__new__(EndpointBank)
    bank.token = 'a' * 48
    bank.guard = threading.Lock()

    @contextlib.contextmanager
    def inactive(parent):
        raise PermissionError('inactive test admission')
        yield

    bank.active = inactive
    request = {'operation': 'allocate', 'parent': 'b' * 48, 'token': bank.token, 'resource': 'browser-endpoints'}
    with pytest.raises(PermissionError, match='inactive'):
        bank.operate(request, None)
    request['token'] = '0' * 48
    with pytest.raises(PermissionError, match='invalid endpoint'):
        bank.operate(request, None)
    request['token'] = bank.token
    request['port'] = 8770
    with pytest.raises(ValueError, match='invalid endpoint operation'):
        bank.operate(request, None)


def test_supervisor_listener_refuses_foreign_and_inactive_leases(bank, monkeypatch):
    monkeypatch.setenv('DEV_TEST_REMOTE_LEASE', 'b' * 48)
    with pytest.raises(RuntimeError, match='transfer failed'):
        Listener()


def test_listener_leases_are_not_reused_before_descendant_cleanup(bank):
    capacity = int(os.environ['DEV_TEST_BROWSER_CAPACITY'])
    assert 1 <= capacity <= 128
    allocated = 0
    for _ in range(capacity + 1):
        try:
            listener = Listener()
        except RuntimeError as error:
            assert 'transfer failed' in str(error)
            break
        listener.close()
        allocated += 1
    else:
        pytest.fail('supervisor listener capacity was not enforced')
    assert allocated <= capacity
    with pytest.raises(RuntimeError, match='transfer failed'):
        Listener()


@pytest.mark.parametrize('address', ['127.0.0.1', '224.0.0.1', '203.0.113.255'])
def test_lan_listener_refuses_unowned_and_loopback_addresses_before_binding(address):
    with pytest.raises((ValueError, PermissionError), match='LAN'):
        local_interface(address)


def test_endpoint_bank_requires_per_active_test_typed_resource_authority():
    EndpointBank.authorized(frozenset({'browser-endpoints'}), 'browser-endpoints')
    with pytest.raises(PermissionError, match='typed resource grant'):
        EndpointBank.authorized(frozenset({'browser-endpoints'}), 'lan-negative')
    with pytest.raises(PermissionError, match='typed resource grant'):
        EndpointBank.authorized(None, 'browser-endpoints')


@pytest.mark.parametrize('proof', [False, None, 'pytest exited'])
def test_browser_run_refuses_endpoint_teardown_without_supervisor_drain_proof(proof):
    from test_browser_admission import BrowserRun

    run = BrowserRun.__new__(BrowserRun)
    retained = object()
    run.bank = retained
    with pytest.raises(RuntimeError, match='retained endpoint bank'):
        run.close(descendants_drained=proof)
    assert run.bank is retained
    run.bank = None
    run.close()
