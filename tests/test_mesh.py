"""Posts are journaled, show a sync status, and reach a paired device through the real exchange."""

import base64
import os
import subprocess
import sys
from pathlib import Path

import pytest
from workspace_kit import Kit, clean_env

from ml_stack import macauth
from ml_stack.fleet.onboard.manifest import Signer
from ml_stack.fleet.remote import Peer
from ml_stack.workspace import journal_merge as rules, mesh_fold, mesh_sync
from ml_stack.workspace.mesh import Mesh
from ml_stack.workspace.service import Workspace

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / 'src'


def kinds(ws):
    return [r['kind'] for r in ws.mesh.journals.rows(ws.mesh.origin)]


def test_with_no_paired_device_every_post_is_journaled_and_synced_without_signing(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    kit.ws.mesh.journals.signer = lambda: pytest.fail('a pool of one never signs')
    alice, bob = kit.agent('alice'), kit.agent('bob')
    kit.ws.send(alice, 'bob', 'status', 'hello')
    kit.ws.announce(alice, 'milestone', 'one line')
    note = kit.ws.note_add(alice, 'fact', 'a title', 'a body')
    assert kinds(kit.ws) == ['message', 'announce', 'note']
    assert note['sync'] == 'synced'
    sent = kit.ws.outbox(alice)
    assert [r['sync'] for r in sent] == ['synced', 'synced']
    assert kit.ws.inbox(bob, raw=True)[0]['raw'] == 'hello'
    assert mesh_sync.sync(kit.ws, []) == []
    assert kit.ws.mesh.journals.vector()[kit.ws.mesh.origin]['seq'] == 3


def test_a_post_with_an_unreachable_peer_is_durable_provisional_and_never_errors(tmp_path, monkeypatch):
    base = clean_env(monkeypatch, tmp_path)
    kit = Kit(base)
    kit.ws.mesh = Mesh(base, signer=Signer.generate, roster=lambda: ['f' * 64])
    key = Signer.generate()
    kit.ws.mesh.journals.signer = lambda: key
    alice, _ = kit.agent('alice'), kit.agent('bob')
    kit.ws.send(alice, 'bob', 'status', 'queued')
    assert [r['sync'] for r in kit.ws.outbox(alice)] == ['provisional']
    nowhere = mesh_sync.Target('f' * 64, Peer('http://127.0.0.1:1', 'x', timeout=1))
    report = mesh_sync.sync(kit.ws, [nowhere])
    assert 'error' in report[0]
    reopened = Workspace(base)
    reopened.mesh = Mesh(base, roster=lambda: ['f' * 64])
    assert [r['sync'] for r in reopened.outbox(alice)] == ['provisional']
    assert reopened.mesh.journals.rows(reopened.mesh.origin)[0]['body']['body'] == 'queued'


def test_a_repeated_request_id_journals_one_row(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    first = kit.ws.mesh.record('message', 'alice', {'body': 'x'}, idem='req-1')
    again = kit.ws.mesh.record('message', 'alice', {'body': 'x'}, idem='req-1')
    other = kit.ws.mesh.record('message', 'bob', {'body': 'x'}, idem='req-1')
    assert again == first and other['seq'] == 2


@pytest.fixture
def pair(tmp_path, monkeypatch):
    """Device A in this process and device B as a real daemon process, each with its own state root."""
    secret = base64.urlsafe_b64encode(b'm' * 32).decode()
    a_for_b, b_for_a = 'a' * 64, 'b' * 64
    root_b = tmp_path / 'device-b'
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(SRC), str(TESTS)]),
                   'ML_STACK_WORKSPACE_HOME': str(root_b / 'workspace'), 'ML_STACK_HOME': str(root_b / 'home'),
                   'PYTHON_KEYRING_BACKEND': 'onboard_support.FileKeyring',
                   'ML_STACK_TEST_KEYRING': str(root_b / 'keyring.json'), 'ML_STACK_NOTIFY': 'off'}
    environment.pop('CLAUDECODE', None)
    server = subprocess.Popen([sys.executable, str(TESTS / 'mesh_server.py'), str(root_b), a_for_b, secret],
                              env=environment, stdout=subprocess.PIPE, text=True)
    try:
        port = int(server.stdout.readline())
        clean_env(monkeypatch, tmp_path / 'device-a')
        kit_a = Kit(tmp_path / 'device-a' / 'workspace')
        key = Signer.generate()
        kit_a.ws.mesh = Mesh(kit_a.base, signer=lambda: key, roster=lambda: [b_for_a])
        kit_b = Kit(root_b / 'workspace')
        kit_b.ws.mesh = Mesh(kit_b.base, roster=lambda: [a_for_b])
        kit_b.ws.mesh.journals.signer = lambda: pytest.fail('the daemon signs for device B')
        peer = Peer(f'http://127.0.0.1:{port}', macauth.derive(base64.urlsafe_b64decode(secret)), timeout=60)
        yield kit_a, kit_b, mesh_sync.Target(b_for_a, peer)
    finally:
        server.terminate()
        server.wait(10)
        server.stdout.close()


def test_two_devices_exchange_messages_announcements_and_notes_and_mark_them_synced(pair):
    a, b, target = pair
    alice = a.agent('alice')
    a.agent('bob')
    bob = b.agent('bob')
    b.agent('alice')
    a.ws.send(alice, 'bob', 'status', 'from device a')
    a.ws.announce(alice, 'milestone', 'a is up')
    a.ws.note_add(alice, 'fact', 'shared fact', 'known on a')
    assert [r['sync'] for r in a.ws.outbox(alice)] == ['provisional', 'provisional']

    report = mesh_sync.sync(a.ws, [target])[0]
    assert report['pushed'] >= 4 and not report['refused'] and report['signed']
    assert [r['sync'] for r in a.ws.outbox(alice)] == ['synced', 'synced']
    assert [m['raw'] for m in b.ws.inbox(bob, raw=True)] == ['from device a']
    senders = {r['from'] for r in b.ws.bus.log.rows()}
    assert senders == {f'alice@{a.ws.mesh.origin[:6]}'}
    found = b.ws.note_search('shared')
    assert [n['author'] for n in found] == [f'alice@{a.ws.mesh.origin[:6]}'] and 'known on a' in found[0]['text']

    b.ws.send(bob, 'alice', 'status', 'from device b')
    assert b.ws.outbox(bob)[0]['sync'] == 'provisional'
    report = mesh_sync.sync(a.ws, [target])[0]
    assert report['pulled'] >= 2 and not report['refused']
    assert [m['raw'] for m in a.ws.inbox(alice, raw=True)] == ['from device b']
    assert b.ws.outbox(bob)[0]['sync'] == 'synced'
    assert mesh_sync.sync(a.ws, [target])[0]['pulled'] == 0
    assert [m['raw'] for m in a.ws.inbox(alice, raw=True)] == ['from device b']

    merged_a = rules.merge({o: a.ws.mesh.journals.rows(o) for o in a.ws.mesh.journals.origins()})
    merged_b = rules.merge({o: b.ws.mesh.journals.rows(o) for o in b.ws.mesh.journals.origins()})
    assert [rules.row_id(r) for r in merged_a] == [rules.row_id(r) for r in merged_b]
    assert mesh_fold.apply(a.ws) == 0


def test_a_peer_serving_a_head_under_another_key_is_refused_and_nothing_of_it_is_folded(pair):
    a, b, target = pair
    alice = a.agent('alice')
    bob = b.agent('bob')
    b.agent('alice')
    b.ws.send(bob, 'alice', 'status', 'real')
    mesh_sync.sync(a.ws, [target])
    other = Signer.generate()
    b_origin = b.ws.mesh.origin
    keys = a.ws.mesh.journals.keys
    keys.put(b_origin, base64.b64encode(other.public).decode())
    b.ws.send(bob, 'alice', 'status', 'after the key was swapped')
    report = mesh_sync.sync(a.ws, [target])[0]
    assert b_origin in report['refused'] and 'different key' in report['refused'][b_origin]
    assert [m['raw'] for m in a.ws.inbox(alice, raw=True)] == ['real']
