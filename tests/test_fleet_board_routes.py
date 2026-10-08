"""Person Board participation and HTTP authorization over a daemon socket."""

import http.client
import json

import pytest
from launch_support import signed
from test_fleet_ui import Serving
from workspace_kit import Kit, clean_env

from ml_stack.fleet import routes
from ml_stack.workspace import tokens

pytestmark = pytest.mark.redteam


@pytest.fixture
def board(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    kit.worker = kit.agent("builder")
    tokens.store(kit.base, tokens.OWNER_FILE, kit.owner)
    server = signed(Serving(tmp_path))
    try:
        yield server, kit
    finally:
        server.close()


def post(server, doc, **options):
    return server.call('/ui/board/post', method='POST', body=doc,
                       headers={'Origin': f'http://127.0.0.1:{server.port}',
                                'Sec-Fetch-Site': 'same-origin', **options.pop('headers', {})}, **options)


def test_local_person_posts_dm_and_thread_without_exposing_credentials(board):
    server, kit = board
    code, directory, _ = server.call('/ui/board/agents')
    assert code == 200 and directory == {'owner_id': 'owner', 'agents': [{'id': 'builder', 'role': 'agent', 'device': kit.ws.registry.info('builder')['device']}]}
    code, dm, _ = post(server, {'to': 'builder', 'body': 'Please inspect the simulation.'})
    assert code == 200
    inbox = kit.ws.inbox(kit.worker)
    assert inbox[0]['from'] == 'owner' and inbox[0]['seq'] == dm['seq']
    code, root, _ = post(server, {'to': '#general', 'body': 'Road simulation', 'subject': 'Driving'})
    assert code == 200
    code, reply, _ = post(server, {'to': '#general', 'body': 'Inspect the sensors.', 'reply_to': root['seq']})
    assert code == 200 and reply['thread'] == root['thread']
    code, thread, _ = server.call(f"/ui/board/thread?root={root['seq']}")
    assert code == 200 and len(thread['messages']) == 2
    assert kit.owner not in json.dumps(thread) and kit.worker not in json.dumps(directory)


def test_joined_session_is_required_before_owner_identity_is_used(board, monkeypatch):
    server, kit = board
    monkeypatch.setattr(routes, 'in_cluster', lambda _: True)
    assert post(server, {'to': 'builder', 'body': 'Unsigned'}, cookie='')[0] == 401
    cookie = server.ui.sessions.cookie_header(server.ui.sessions.open('person', 'launch-ticket'))
    assert post(server, {'to': 'builder', 'body': 'Signed'}, cookie=cookie)[0] == 200
    assert len(kit.ws.inbox(kit.worker)) == 1


@pytest.mark.parametrize('attack', [
    {'headers': {'Origin': 'https://foreign.example'}},
    {'headers': {'Sec-Fetch-Site': 'cross-site'}},
    {'ui_header': False}, {'headers': {'Host': 'foreign.example'}},
])
def test_foreign_browser_cannot_send_as_person(board, attack):
    server, kit = board
    assert post(server, {'to': 'builder', 'body': 'Hostile'}, **attack)[0] in (403, 421)
    assert kit.ws.inbox(kit.worker) == []


@pytest.mark.parametrize('doc', [None, [], {'to': 'builder', 'body': 'x', 'sender': 'builder'},
    {'to': 'builder', 'body': 'x', 'token': 'browser-token'},
    {'to': 'builder', 'body': 'x', 'reply_to': True}, {'to': ['builder'], 'body': 'x'}])
def test_malformed_or_impersonating_post_does_not_deliver(board, doc):
    server, kit = board
    assert post(server, doc)[0] == 400
    assert kit.ws.inbox(kit.worker) == []


def test_page_cannot_post_announcements_or_use_agent_owner_file(board):
    server, kit = board
    assert post(server, {'to': '#announcements', 'body': 'Announcement'})[0] == 403
    tokens.store(kit.base, tokens.OWNER_FILE, kit.worker)
    assert post(server, {'to': '#general', 'body': 'Agent impersonation'})[0] == 403


def test_oversized_post_and_unsupported_methods_do_not_mutate(board):
    server, kit = board
    assert post(server, {'to': 'builder', 'body': 'x' * 33000})[0] == 413
    assert server.call('/ui/board/post', method='DELETE')[0] == 405
    conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=5)
    try:
        conn.request('POST', '/ui/board/post', body=b'\xff', headers={
            'X-ML-Stack-UI': '1', 'Content-Type': 'application/json', 'Cookie': server.cookie,
            'Origin': f'http://127.0.0.1:{server.port}'})
        assert conn.getresponse().status == 400
    finally:
        conn.close()
    assert kit.ws.inbox(kit.worker) == []


def test_ack_marks_only_an_existing_visible_message_in_its_scope(board):
    server, kit = board
    _, sent, _ = post(server, {'to': 'builder', 'body': 'Inspect this run.'})
    headers = {'Origin': f'http://127.0.0.1:{server.port}', 'Sec-Fetch-Site': 'same-origin'}
    def ack(doc):
        return server.call('/ui/board/ack', method='POST', body=doc, headers=headers)[0]
    assert ack({'to': 'builder', 'through': sent['seq'] + 100}) == 403
    assert ack({'board': '#general', 'through': sent['seq']}) == 403
    assert ack({'to': 'builder', 'through': True}) == 400
    assert kit.ws.board.store.marks('owner') == {}
    assert ack({'to': 'builder', 'through': sent['seq']}) == 200
    assert kit.ws.board.store.marks('owner') == {'dm:builder': sent['seq']}


def test_board_extension_is_registered_in_installed_distribution_metadata(board):
    from importlib.metadata import distribution

    server, _ = board
    entries = [entry for entry in distribution('ml-stack').entry_points
               if entry.group == 'ml_stack.ui_routes' and entry.name == 'board']
    assert len(entries) == 1
    assert entries[0].value == 'ml_stack.workspace.fleet_routes:route'
    assert callable(entries[0].load())
    assert server.call('/ui/board/boards')[0] == 200
