"""Authenticated local person access to project conversations."""

import io
import json
from types import SimpleNamespace

import pytest

from ml_stack.fleet.project_board_routes import ProjectBoardRoutes
from ml_stack.fleet.routes import Base
from ml_stack.fleet.session import Sessions
from ml_stack.workspace import tokens
from ml_stack.workspace.remote_host import WorkspaceHost
from ml_stack.workspace.service import Workspace

PROJECT = "a" * 32
OTHER = "b" * 32


class Request(ProjectBoardRoutes, Base):
    def send(self, status, payload, extra=None):
        self.handler.send_response(status)
        self.handler.wfile.write(json.dumps(payload).encode())


@pytest.fixture
def project_board(tmp_path):
    workspaces = {}
    for ident in (PROJECT, OTHER):
        ws = Workspace(tmp_path / ident)
        owner = ws.init("demo-owner")
        tokens.store(ws.base, tokens.OWNER_FILE, owner)
        agent = ws.mint(owner, "demo-worker")
        ws.send(agent, "#general", "note", ident)
        workspaces[ident] = ws
    projects = {ident: SimpleNamespace(name="fixture", board_host="http://127.0.0.1:8770")
                for ident in workspaces}
    def get(ident):
        if ident not in projects:
            raise ValueError("Project is not registered here")
        return projects[ident]
    registry = SimpleNamespace(machine="local", hosts=lambda host: host == "http://127.0.0.1:8770", get=get,
                               workspace_base=lambda ident: workspaces[ident].base)
    sessions = Sessions()
    session = sessions.open("fixture-person", "launch-ticket")
    cookie_value = f"ml_stack_ui={session.sid}"
    ui = SimpleNamespace(sessions=sessions, projects=registry, record=lambda *a, **k: None,
                         workspaces=WorkspaceHost(registry),
                         authed=lambda cookie: cookie == cookie_value, credentialed=lambda cookie: cookie == cookie_value, host_ok=lambda host: host == "127.0.0.1:8770")
    def call(suffix, *, project=PROJECT, method="GET", body=None, **options):
        cookie = options.get("cookie", cookie_value)
        ip = options.get("ip", "127.0.0.1")
        origin = options.get("origin", "http://127.0.0.1:8770")
        raw = options.get("raw", json.dumps(body).encode() if body is not None else b"")
        headers = {"Host": "127.0.0.1:8770", "Cookie": cookie, "Origin": origin,
                   "Content-Type": "application/json", "Content-Length": str(len(raw)),
                   **options.get("headers", {})}
        codes = []
        handler = SimpleNamespace(path=options.get("path", f"/ui/projects/{project}/board/{suffix}"), command=method,
                                  client_address=(ip, 1000), headers=headers,
                                  rfile=io.BytesIO(raw), wfile=io.BytesIO(),
                                  server=SimpleNamespace(server_address=("127.0.0.1", 8770)),
                                  send_response=codes.append, send_header=lambda *args: None,
                                  end_headers=lambda: None)
        assert Request(ui, handler).route()
        return codes[0], json.loads(handler.wfile.getvalue())
    for ident in workspaces:
        assert call("connect", project=ident, method="POST", body={})[0] == 200
    return call, workspaces, projects


def test_selected_project_messages_and_person_post(project_board):
    call, workspaces, _ = project_board
    status, result = call("messages?board=%23general")
    assert status == 200
    assert [row["body"] for row in result["messages"]] == [PROJECT]
    status, posted = call("post", method="POST", body={"to": "#general", "body": "person message"})
    assert status == 200 and posted["to"] == "#general"
    status, result = call("messages?board=%23general")
    assert result["messages"][-1]["body"] == "person message"
    assert result["messages"][-1]["from"] == "demo-owner"
    assert len(workspaces[OTHER].board._rows()) == 1


@pytest.mark.parametrize("kwargs", [{"cookie": ""}, {"ip": "192.0.2.8"},
                                   {"method": "POST", "origin": "http://evil.test",
                                    "body": {"to": "#general", "body": "hostile"}}])
def test_person_board_rejects_foreign_requests(project_board, kwargs):
    call, _, _ = project_board
    assert call("post" if kwargs.get("method") == "POST" else "boards", **kwargs)[0] == 403


def test_missing_remote_or_unconfigured_project_never_falls_back(project_board):
    call, _, projects = project_board
    assert call("boards", project="c" * 32)[0] == 409
    projects[PROJECT].board_host = "http://foreign:8770"
    assert call("boards")[0] == 409
    projects[PROJECT].board_host = ""
    assert call("boards")[0] == 409


def test_person_identity_is_required_and_never_created(project_board):
    call, workspaces, _ = project_board
    ws = workspaces[PROJECT]
    owner_path = tokens.directory(ws.base) / tokens.OWNER_FILE
    owner_path.unlink()
    assert call("boards")[0] == 200
    agents = ws.registry._load()
    agents['demo-owner'].pop('person_project')
    ws.registry._save(agents)
    assert call("boards")[0] == 503
    assert 'person_project' not in ws.registry._load()['demo-owner']
    assert not owner_path.exists()


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH"])
def test_person_board_refuses_unsupported_methods(project_board, method):
    assert project_board[0]("boards", method=method)[0] == 405


def test_person_threads_direct_messages_and_ack(project_board):
    call, workspaces, _ = project_board
    status, root = call("post", method="POST", body={"to": "#general", "body": "root"})
    assert status == 200
    status, _reply = call("post", method="POST", body={"to": "#general", "body": "reply", "reply_to": root["seq"]})
    assert status == 200
    status, thread = call(f"thread?root={root['seq']}")
    assert status == 200
    assert {row["body"] for row in thread["messages"]} == {"root", "reply"}
    status, sent = call("post", method="POST", body={"to": "demo-worker", "body": "direct"})
    assert status == 200
    status, direct = call("dm?a=demo-owner&b=demo-worker")
    assert status == 200 and direct["messages"][-1]["body"] == "direct"
    status, ack = call("ack", method="POST", body={"to": "demo-worker", "through": sent["seq"]})
    assert status == 200 and ack["through"] == sent["seq"]
    assert workspaces[PROJECT].board.store.marks("demo-owner")["dm:demo-worker"] == sent["seq"]


@pytest.mark.parametrize('scheme', ['http', 'https'])
def test_person_post_accepts_exact_browser_origin(project_board, scheme):
    status, sent = project_board[0]('post', method='POST', origin=f'{scheme}://127.0.0.1:8770',
                                body={'to': '#general', 'body': 'browser message'})
    assert status == 200 and sent['to'] == '#general'


@pytest.mark.parametrize('origin', ['https://127.0.0.1:8771', 'https://localhost:8770',
                                   'https://evil.test:8770', 'https://127.0.0.1:8770/'])
def test_person_post_rejects_different_browser_origin(project_board, origin):
    assert project_board[0]('post', method='POST', origin=origin,
                         body={'to': '#general', 'body': 'foreign message'})[0] == 403


def test_person_task_create_rejects_other_project(project_board):
    call, _, _ = project_board
    path = f'/ui/projects/{PROJECT}/tasks'
    status, task = call('', path=path, method='POST', body={
        'spec': {'title': 'Replay', 'acceptance': ['Replay passes']}})
    assert status == 200 and task['project']['key'] == PROJECT
    status, _ = call('', path=path, method='POST', body={
        'spec': {'title': 'Foreign', 'acceptance': ['Denied'], 'project': {'key': OTHER}}})
    assert status == 403


@pytest.mark.parametrize(('length', 'expected'), [('-1', 403), ('NaN', 403), (' 2', 400),
                                               ('000000002', 400), ('1025', 400),
                                               ('99999999', 403), ('\u0661', 403)])
def test_person_connect_rejects_invalid_content_length_without_registry_mutation(project_board, length, expected):
    call, workspaces, _ = project_board
    ws = workspaces[PROJECT]
    agents = ws.registry._load()
    agents['demo-owner'].pop('person_project')
    ws.registry._save(agents)
    original = ws.registry.path.read_bytes()
    status, _ = call('connect', method='POST', raw=b'{}', headers={'Content-Length': length})
    assert status == expected
    assert ws.registry.path.read_bytes() == original


@pytest.mark.parametrize('raw', [b'{', b'', b'null', b'[]', b'"person"', b'{"role":"human"}',
                               b'{"project_id":"foreign"}'])
def test_person_connect_rejects_invalid_json_without_registry_mutation(project_board, raw):
    call, workspaces, _ = project_board
    ws = workspaces[PROJECT]
    agents = ws.registry._load()
    agents['demo-owner'].pop('person_project')
    ws.registry._save(agents)
    original = ws.registry.path.read_bytes()
    status, _ = call('connect', method='POST', raw=raw)
    assert status == 400
    assert ws.registry.path.read_bytes() == original
