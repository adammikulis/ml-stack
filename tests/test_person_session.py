"""Browser person bindings and request-scoped authority."""

import io
from types import SimpleNamespace

import pytest

from ml_stack.fleet.session import Sessions
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.person_session import SessionWorkspace, SetupRequired
from ml_stack.workspace.remote_host import WorkspaceHost
from ml_stack.workspace.remote_protocol import METHODS
from ml_stack.workspace.service import Workspace

PROJECT = 'a' * 32


@pytest.fixture
def bound(tmp_path):
    project = SimpleNamespace(board_host='https://127.0.0.1:8770')
    projects = SimpleNamespace(machine='local', hosts=lambda host: host == 'https://127.0.0.1:8770', get=lambda ident: project,
                               workspace_base=lambda ident: tmp_path / ident)
    host = WorkspaceHost(projects)
    sessions = Sessions()
    session = sessions.open('fixture-person')
    cookie = f'ml_stack_ui={session.sid}'
    ui = SimpleNamespace(sessions=sessions, projects=projects, workspaces=host,
                         authed=lambda value: sessions.get(value.split('=', 1)[-1]) is not None,
                         host_ok=lambda value: value == '127.0.0.1:8770')
    request = SimpleNamespace(ui=ui, cookie=cookie, client_ip='127.0.0.1',
                              host_header='127.0.0.1:8770', method='POST',
                              path=f'/ui/projects/{PROJECT}/board/connect',
                              handler=SimpleNamespace(headers={'Host': '127.0.0.1:8770',
                                                               'Origin': 'https://127.0.0.1:8770',
                                                               'Content-Type': 'application/json',
                                                               'Content-Length': '2'},
                                                      rfile=io.BytesIO(b'{}'),
                                                      server=SimpleNamespace(server_address=('127.0.0.1', 8770))))
    return host, request, project, session


def test_explicit_connect_creates_hashless_person_without_token(bound):
    host, request, _, _ = bound
    ws = SessionWorkspace(host, request, PROJECT)
    with pytest.raises(SetupRequired):
        ws.auth(ws._actor)
    assert ws.registry.ids() == []
    assert ws.connect()['me'] == 'local-person'
    entry = ws.registry._load()['local-person']
    assert 'hash' not in entry and 'token' not in entry
    assert ws.auth(ws._actor).can == ('read', 'send')
    with pytest.raises(Denied):
        ws.registry.authenticate('mlws1.local-person.forged')
    with pytest.raises(Denied):
        Workspace(ws.base).auth(ws._actor)
    with pytest.raises(Denied):
        SessionWorkspace(host, request, PROJECT).auth(ws._actor)
    with pytest.raises(Denied):
        ws.mint(ws._actor, 'helper')
    assert 'connect' not in METHODS


@pytest.mark.parametrize('change', ['session', 'expiry', 'authority', 'origin', 'host', 'ip', 'project', 'person'])
def test_actor_rechecks_live_session_project_and_person(bound, change):
    host, request, project, session = bound
    ws = SessionWorkspace(host, request, PROJECT)
    ws.connect()
    if change == 'session':
        request.ui.sessions.close(session.sid)
    elif change == 'expiry':
        session.expires_at = 0
    elif change == 'authority':
        project.board_host = 'https://foreign:8770'
    elif change == 'origin':
        request.handler.headers['Origin'] = 'https://evil.test'
    elif change == 'host':
        request.host_header = 'evil.test'
    elif change == 'ip':
        request.client_ip = '192.0.2.8'
    elif change == 'project':
        request.path = f'/ui/projects/{"b" * 32}/board/post'
    else:
        agents = ws.registry._load()
        agents['local-person']['revoked'] = True
        ws.registry._save(agents)
    with pytest.raises(Denied):
        ws.auth(ws._actor)


def test_connect_refuses_get_collision_and_revoked_person(bound):
    host, request, _, _ = bound
    ws = SessionWorkspace(host, request, PROJECT)
    request.method = 'GET'
    with pytest.raises(Denied):
        ws.connect()
    request.method = 'POST'
    ws.registry._save({'local-person': {'role': 'agent', 'revoked': False, 'expires': 0}})
    with pytest.raises(Denied):
        ws.connect()
    ws.registry._save({'local-person': {'role': 'human', 'revoked': True, 'expires': 0}})
    with pytest.raises(Denied):
        ws.connect()


def test_task_write_revalidates_person_and_preserves_independent_review(bound):
    from ml_stack.workspace.taskboard import TaskBoard

    host, request, _, session = bound
    ws = SessionWorkspace(host, request, PROJECT)
    ws.connect()
    board = TaskBoard(ws)
    task = board.create(ws._actor, {'title': 'Replay', 'acceptance': ['Replay passes']})
    assert task['created_by'] == 'local-person'
    with pytest.raises(Denied):
        board._reviewer(ws.auth(ws._actor), {**task, 'worker': 'local-person'})
    request.ui.sessions.close(session.sid)
    with pytest.raises(Denied):
        board.create(ws._actor, {'title': 'Revoked', 'acceptance': ['Denied']})


def test_session_person_preserves_existing_capability_restrictions(bound):
    host, request, _, _ = bound
    ws = SessionWorkspace(host, request, PROJECT)
    ws.connect()
    agents = ws.registry._load()
    agents['local-person']['can'] = ['read']
    ws.registry._save(agents)
    assert ws.auth(ws._actor).can == ('read',)
    with pytest.raises(Denied):
        ws.send(ws._actor, '#general', 'note', 'restricted')
