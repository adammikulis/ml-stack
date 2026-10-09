"""Invitation parsing, pinned transport and authenticated membership grants."""
import hashlib
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ml_stack.fleet import invite_client, tls
from ml_stack.fleet.discovery import DiscoveryError, Membership
from ml_stack.fleet.invites import Invitations, decode, encode


def payload(**changes):
    data = {'v': 1, 'endpoint': 'https://192.168.2.59:8770', 'fingerprint': 'a' * 64,
                'id': 'b' * 32, 'secret': encode(b's' * 32), 'expires': 1600, 'kind': 'computer'}
    data.update(changes)
    return 'ml-stack://enroll?data=' + encode(json.dumps(data).encode())


@pytest.fixture
def lan_dns(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *_args, **_kwargs:
        [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('192.168.2.59', 8770))])


def test_current_computer_invite_is_accepted_without_contacting_host(lan_dns):
    assert invite_client.parse_invite(payload(), now=1000)['id'] == 'b' * 32


@pytest.mark.parametrize('changes', [{'v': True}, {'kind': 'android'}, {'expires': 1000},
    {'expires': 1601}, {'secret': '+' * 43}, {'fingerprint': 'A' * 64}, {'id': []},
    {'endpoint': 'http://192.168.2.59:8770'}, {'endpoint': 'https://user@192.168.2.59:8770'},
    {'endpoint': 'https://192.168.2.59:8770/redirect'}, {'endpoint': 'https://192.168.2.59:8770?next=x'}])
def test_malformed_invites_are_refused(lan_dns, changes):
    with pytest.raises(ValueError):
        invite_client.parse_invite(payload(**changes), now=1000)


def test_duplicate_json_fields_are_not_accepted(lan_dns):
    raw = decode(payload().split('data=')[1]).decode().replace('"v": 1', '"v": 1, "v": 1')
    with pytest.raises(ValueError, match='duplicate'):
        invite_client.parse_invite('ml-stack://enroll?data=' + encode(raw.encode()), now=1000)


@pytest.mark.parametrize('address', ['8.8.8.8', '127.0.0.1', '0.0.0.0', '224.0.0.1'])  # noqa: S104
def test_every_resolved_address_must_be_a_remote_lan_device(monkeypatch, address):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *_args, **_kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('192.168.2.59', 8770)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 8770))])
    with pytest.raises((ValueError, OSError)):
        invite_client.parse_invite(payload(), now=1000)


@pytest.fixture(params=["dev", "prod"])
def invitation_server(tmp_path, monkeypatch, request):
    identity = tls.identity(tmp_path / 'tls', 'test-device')
    member = Membership(group='Test cluster', key=encode(b'k' * 32).encode(),
                        mode=request.param, selection='manual')
    hits = []
    source = [None]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            hits.append(self.path)
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            try:
                reply = source[0].exchange(self.path.rsplit('/', 1)[1], body)
                status = 200
            except ValueError:
                reply, status = {'error': 'refused'}, 400
            raw = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.socket = tls.server_context(identity).wrap_socket(server.socket, server_side=True)
    endpoint = f'https://192.168.2.59:{server.server_port}'
    enrolled = []
    store = Invitations(lambda: [member], lambda: (endpoint, identity.fingerprint),
                        enrol=lambda *row: enrolled.append(row))
    store.enrolled, store.identity = enrolled, identity
    source[0] = store
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(invite_client, '_addresses', lambda url: (
        invite_client.urlsplit(url), [(socket.AF_INET, socket.SOCK_STREAM, 6, '',
                                      ('127.0.0.1', server.server_port))]))
    try:
        yield store, member, hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_actual_pinned_tls_exchange_joins_once(invitation_server):
    store, member, hits = invitation_server
    invite = store.mint(member.group)['invite']
    redeemed, host_cert = invite_client.redeem(invite, 'test-device')
    assert host_cert == store.identity.beacon
    assert store.enrolled == [(member.group, tls.local().beacon, 'test-device')]
    assert redeemed == member
    assert redeemed.mode == member.mode
    assert redeemed.selection == 'manual'
    assert hits == ['/join/invite/challenge', '/join/invite/redeem']
    with pytest.raises(ValueError, match='refused'):
        invite_client.redeem(invite, 'test-device')


def test_wrong_certificate_is_refused_before_any_http(invitation_server):
    store, member, hits = invitation_server
    data = json.loads(decode(store.mint(member.group)['invite'].split('data=')[1]))
    data['fingerprint'] = hashlib.sha256(b'wrong-certificate').hexdigest()
    invite = 'ml-stack://enroll?data=' + encode(json.dumps(data).encode())
    with pytest.raises(ValueError, match='certificate'):
        invite_client.redeem(invite, 'test-device')
    assert hits == []


def test_modified_grant_is_rejected(invitation_server, monkeypatch):
    store, member, _ = invitation_server
    post = invite_client._post

    def tampered(data, path, fields):
        answer, certificate = post(data, path, fields)
        if path.endswith('/redeem'):
            answer['grant_data'] = encode(json.dumps({'kind': 'computer', 'group': 'Wrong cluster',
                                                         'key': member.key.decode()}).encode())
        return answer, certificate

    monkeypatch.setattr(invite_client, '_post', tampered)
    with pytest.raises(ValueError, match='authenticated'):
        invite_client.redeem(store.mint(member.group)['invite'], 'test-device')


@pytest.mark.redteam
@pytest.mark.parametrize("mode", [None, "invalid", 1, []])
def test_authenticated_grant_refuses_missing_or_invalid_mode(invitation_server, monkeypatch, mode):
    store, member, _ = invitation_server
    grant = store._grant

    def malformed(row, fields, held):
        payload = grant(row, fields, held)
        if mode is None:
            payload.pop("mode")
        else:
            payload["mode"] = mode
        return payload

    monkeypatch.setattr(store, "_grant", malformed)
    with pytest.raises((ValueError, DiscoveryError)):
        invite_client.redeem(store.mint(member.group)["invite"], "test-device")
