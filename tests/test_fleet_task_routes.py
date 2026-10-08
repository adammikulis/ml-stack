"""Actual Fleet sockets enforce person authority before task mutations."""

from types import SimpleNamespace

import pytest
from test_fleet_ui import Serving
from workspace_kit import Kit, clean_env

from ml_stack.fleet import extension_routes, routes
from ml_stack.workspace import task_routes, tokens
from ml_stack.workspace.taskboard import TaskBoard

pytestmark = pytest.mark.redteam


@pytest.fixture
def tasks(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    tokens.store(kit.base, tokens.OWNER_FILE, kit.owner)
    entry = SimpleNamespace(name='tasks', load=lambda: task_routes.route)
    monkeypatch.setattr(extension_routes, 'entry_points', lambda **kw: [entry])
    server = Serving(tmp_path)
    try:
        yield server, kit
    finally:
        server.close()


def post(server, body, **options):
    return server.call('/ui/tasks', method='POST', body=body,
                       headers={'Origin': f'http://127.0.0.1:{server.port}',
                                **options.pop('headers', {})}, **options)


def test_person_creates_native_graph_task_without_browser_credentials(tasks):
    server, kit = tasks
    code, task, _ = post(server, {'spec': {'title': 'Replay', 'acceptance': ['Replay passes']}})
    assert code == 200 and task['created_by'] == 'owner'
    assert TaskBoard(kit.ws).get(kit.owner, task['id'])['state'] == 'queued'
    code, result, _ = server.call('/ui/tasks')
    assert code == 200 and len(result['tasks']) == 1
    assert kit.owner not in str(result)


@pytest.mark.parametrize('options', [{'ui_header': False},
    {'headers': {'Origin': 'https://foreign.example'}},
    {'headers': {'Sec-Fetch-Site': 'cross-site'}}])
def test_foreign_page_cannot_create_tasks(tasks, options):
    server, kit = tasks
    assert post(server, {'spec': {'title': 'Hostile', 'acceptance': ['x']}}, **options)[0] == 403
    assert TaskBoard(kit.ws).list(kit.owner)['tasks'] == []


def test_joined_unsigned_and_browser_claims_are_refused(tasks, monkeypatch):
    server, kit = tasks
    monkeypatch.setattr(routes, 'in_cluster', lambda _: True)
    assert post(server, {'spec': {'title': 'x', 'acceptance': ['x']}})[0] == 401
    cookie = server.ui.sessions.cookie_header(server.ui.sessions.open('person'))
    assert post(server, {'action': 'claim', 'token': 'worker', 'allocation_id': 'fake'}, cookie=cookie)[0] == 400
    assert TaskBoard(kit.ws).list(kit.owner)['tasks'] == []


@pytest.mark.parametrize('body', [[], None, {'action': 'create', 'spec': None},
    {'action': 'review', 'id': '../escape', 'decision': {}}, {'spec': {}, 'identity': 'worker'}])
def test_malformed_requests_and_identity_overrides_do_not_mutate(tasks, body):
    server, kit = tasks
    assert post(server, body)[0] == 400
    assert server.call('/ui/tasks', method='DELETE')[0] == 405
    assert TaskBoard(kit.ws).list(kit.owner)['tasks'] == []
