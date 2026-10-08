"""Authenticated graph inspection through Fleet."""

from urllib.parse import urlencode

import pytest
from test_fleet_ui import Serving

from ml_stack.graph.store import GraphStore


@pytest.fixture
def daemon(tmp_path):
    server = Serving(tmp_path)
    with GraphStore(server.files / 'knowledge.db') as graph:
        graph.upsert_node({'id': 'model:alpha', 'kind': 'model', 'label': 'Alpha model',
                           'attrs': {'precision': '4bit'}, 'evidence': ['source:one']})
        graph.upsert_node({'id': 'task:beta', 'kind': 'task', 'label': 'Beta training'})
        graph.upsert_edge({'source': 'model:alpha', 'target': 'task:beta',
                           'rel': 'trained_for', 'weight': 3})
    try:
        yield server
    finally:
        server.close()


def test_inspector_search_details_and_relations(daemon):
    code, data, _ = daemon.call('/ui/knowledge/stores')
    assert code == 200, data
    assert {'path': 'knowledge.db', 'name': 'knowledge.db', 'directory': False} in data['entries']
    code, data, _ = daemon.call('/ui/knowledge/nodes?' + urlencode({
        'store': 'knowledge.db', 'q': 'ALPHA', 'kind': 'model'}))
    assert code == 200, data
    assert data['counts']['nodes'] == 2
    assert [node['id'] for node in data['nodes']] == ['model:alpha']
    assert not data['more']
    code, data, _ = daemon.call('/ui/knowledge/node?' + urlencode({
        'store': 'knowledge.db', 'id': 'model:alpha'}))
    assert code == 200, data
    assert data['node']['attrs'] == {'precision': '4bit'}
    assert data['node']['details'] == {'evidence': ['source:one']}
    assert data['relations'][0]['target'] == 'task:beta'
    assert data['relations'][0]['rel'] == 'trained_for'
    code, data, _ = daemon.call('/ui/knowledge/node?store=knowledge.db&id=task:beta')
    assert data['relations'][0]['source'] == 'model:alpha'
    assert daemon.call('/ui/knowledge/node?store=knowledge.db&id=absent')[0] == 404


def test_inspector_authorization_path_bounds_and_read_only(daemon, tmp_path):
    assert daemon.call('/ui/knowledge/stores', ui_header=False)[0] == 403
    assert daemon.call('/ui/knowledge/nodes?store=../private.db')[0] == 400
    assert daemon.call('/ui/knowledge/nodes?store=missing.db')[0] == 400
    assert daemon.call('/ui/knowledge/nodes?store=knowledge.db&offset=-1')[0] == 400
    assert daemon.call('/ui/knowledge/nodes?store=knowledge.db&offset=1000001')[0] == 400
    assert daemon.call('/ui/knowledge/stores', method='POST', body={})[0] == 405
    (daemon.files / 'outside.db').symlink_to(tmp_path / 'private.db')
    assert daemon.call('/ui/knowledge/nodes?store=outside.db')[0] == 400
    code, data, _ = daemon.call('/ui/knowledge/stores')
    assert code == 200, data
    assert all(entry['name'] != 'outside.db' for entry in data['entries'])
    with GraphStore(daemon.files / 'knowledge.db', read_only=True) as graph:
        assert graph.counts()['nodes'] == 2


def test_inspector_pagination_filters_before_limit(daemon):
    with GraphStore(daemon.files / 'knowledge.db') as graph:
        for i in range(70):
            graph.upsert_node({'id': f'node:{i:03}', 'kind': 'document', 'label': f'Document {i:03}'})
    code, data, _ = daemon.call('/ui/knowledge/nodes?store=knowledge.db&kind=document')
    assert code == 200, data
    assert len(data['nodes']) == 60 and data['more']
    code, data, _ = daemon.call('/ui/knowledge/nodes?store=knowledge.db&kind=document&offset=60')
    assert len(data['nodes']) == 10 and not data['more']
    code, data, _ = daemon.call('/ui/knowledge/nodes?store=knowledge.db&q=Document+069')
    assert [node['id'] for node in data['nodes']] == ['node:069']


@pytest.mark.slow
def test_graph_browser_navigation_search_and_neighbor(daemon, tmp_path, playwright):
    from playwright.sync_api import expect

    from ml_stack.scrape.browser import Window, browser

    daemon.ui.settings.setup_done = True
    with browser(Window(profile=tmp_path / 'browser', headless=True), play=playwright) as page:
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{daemon.port}/ui')
        page.locator('fleet-nav a[href="#knowledge"]').click()
        page.get_by_role('button', name='Graph · knowledge.db', exact=True).click()
        page.get_by_label('Search nodes').fill('Alpha')
        page.locator('knowledge-view').get_by_role('button', name='Search', exact=True).click()
        page.get_by_role('button', name='Alpha model · model', exact=True).click()
        detail = page.locator('knowledge-view #detail')
        expect(detail.get_by_role('heading', name='Alpha model', exact=True)).to_be_visible()
        expect(detail).to_contain_text('precision')
        expect(detail).to_contain_text('trained_for')
        palette = page.evaluate("""() => {
            const root = document.documentElement;
            root.style.setProperty('--poolside-yellow', '#ffd166');
            root.style.setProperty('--poolside-yellow-ink', '#000000');
            const label = document.querySelector('knowledge-view .graph-node-label');
            const caption = document.querySelector('knowledge-view .graph-node-caption');
            const light = [getComputedStyle(label).fill, getComputedStyle(caption).fill];
            root.style.setProperty('--poolside-yellow', '#101010');
            root.style.setProperty('--poolside-yellow-ink', '#ffffff');
            const custom = [getComputedStyle(label).fill, getComputedStyle(caption).fill];
            root.style.removeProperty('--poolside-yellow');
            root.style.removeProperty('--poolside-yellow-ink');
            return {light, custom};
        }""")
        assert palette == {'light': ['rgb(0, 0, 0)', 'rgb(255, 209, 102)'],
                           'custom': ['rgb(255, 255, 255)', 'rgb(16, 16, 16)']}
        expect(detail.locator('details')).not_to_have_attribute('open', '')
        detail.get_by_role('button', name='Inspect Beta training', exact=True).press('Enter')
        expect(detail.get_by_role('heading', name='Beta training', exact=True)).to_be_visible()
        detail.get_by_role('button', name='Alpha model', exact=True).click()
        expect(detail.get_by_role('heading', name='Alpha model', exact=True)).to_be_visible()
        page.get_by_label('Node kind').select_option('task')
        expect(page.locator('knowledge-view #nodes')).to_contain_text('No nodes match')
        race = page.evaluate("""async () => {
            const view = document.querySelector('knowledge-view');
            const api = window.workspaceModel.api;
            let rejectOld;
            window.workspaceModel.api = (path, options) => path.includes('q=old-request')
                ? new Promise((_resolve, reject) => { rejectOld = reject; }) : api(path, options);
            view.search.value = 'old-request';
            const old = view.loadNodes();
            await view.selectStore('knowledge.db');
            rejectOld(new Error('Stale search error'));
            await old;
            window.workspaceModel.api = api;
            return {note:view.note.textContent, busy:view.querySelector('#graph-panel').hasAttribute('aria-busy')};
        }""")
        assert race == {'note': '', 'busy': False}
        folder_race = page.evaluate("""async () => {
            const view = document.querySelector('knowledge-view');
            const api = window.workspaceModel.api;
            let rejectOld;
            window.workspaceModel.api = path => new Promise((_resolve, reject) => { rejectOld = reject; });
            const old = view.loadStores();
            window.workspaceModel.api = api;
            await view.loadStores();
            rejectOld(new Error('Stale folder error'));
            await old;
            return view.note.textContent;
        }""")
        assert folder_race == ''
        assert not errors


def test_registered_conversation_graph_without_files_root(daemon, tmp_path):
    from ml_stack.fleet.conversations import Conversations

    daemon.ui.conversations = Conversations(tmp_path / 'chat')
    chat = daemon.ui.conversations.start(model='local-model', title='Experiment notes')
    daemon.ui.conversations.append(chat.id, 'user', 'Inspect training results')
    daemon.ui.runner = None
    code, data, _ = daemon.call('/ui/knowledge/stores')
    assert code == 200, data
    assert data['sources'] == [{'path': 'app:conversations', 'name': 'Conversations and models'}]
    assert not data['files_available']
    code, data, _ = daemon.call('/ui/knowledge/nodes?store=app:conversations&kind=conversation')
    assert code == 200, data
    assert data['nodes'][0]['label'] == 'Experiment notes'
    code, data, _ = daemon.call('/ui/knowledge/node?' + urlencode({
        'store': 'app:conversations', 'id': 'conversation:' + chat.id}))
    assert code == 200, data
    assert {edge['rel'] for edge in data['relations']} == {'contains', 'uses-model'}
    assert daemon.call('/ui/knowledge/nodes?store=app:unknown')[0] == 403
    assert daemon.call('/ui/knowledge/stores', headers={'Origin': 'https://other.invalid'})[0] == 403


def test_registered_project_graph_requires_local_person(daemon, tmp_path):
    import json
    from dataclasses import asdict

    from ml_stack.fleet.projects import Project, ProjectRegistry

    project_id = 'a' * 32
    base = tmp_path / 'projects'
    base.mkdir()
    project = Project(id=project_id, name='Experiment', root=str(tmp_path / 'checkout'),
                      source_machine='local', board_host='https://local:8770')
    (base / 'projects.json').write_text(json.dumps({'projects': [asdict(project)]}))
    daemon.ui.projects = ProjectRegistry(base, 'local', host='https://local:8770')
    workspace = daemon.ui.projects.workspace_base(project_id)
    workspace.mkdir(parents=True)
    with GraphStore(workspace / 'coordination.db') as graph:
        graph.upsert_node({'id': 'task:one', 'kind': 'task', 'label': 'Evaluate policy'})
    source = 'project:' + project_id + ':coordination'
    assert daemon.call('/ui/knowledge/nodes?' + urlencode({'store': source}))[0] == 403
    session = daemon.ui.sessions.open()
    cookie = 'ml_stack_ui=' + session.sid
    code, data, _ = daemon.call('/ui/knowledge/stores', cookie=cookie)
    assert code == 200, data
    assert data['sources'] == [{'path': source, 'name': 'Experiment · tasks and coordination'}]
    code, data, _ = daemon.call('/ui/knowledge/nodes?' + urlencode({'store': source}), cookie=cookie)
    assert code == 200, data
    assert data['nodes'][0]['id'] == 'task:one'
    assert daemon.call('/ui/knowledge/nodes?' + urlencode({
        'store': 'project:' + 'b' * 32 + ':coordination'}), cookie=cookie)[0] == 403


def test_large_fields_and_directory_discovery_are_bounded(daemon):
    with GraphStore(daemon.files / 'knowledge.db') as graph:
        graph.upsert_node({'id': 'large', 'label': 'Long evidence', 'attrs': {'body': 'x' * 40_000}})
    code, data, _ = daemon.call('/ui/knowledge/node?store=knowledge.db&id=large')
    assert code == 200, data
    assert data['node']['attrs']['truncated']
    assert len(data['node']['attrs']['preview']) == 32_768
    for i in range(201):
        (daemon.files / f'folder-{i}').mkdir()
    code, data, _ = daemon.call('/ui/knowledge/stores')
    assert code == 200, data
    assert data['truncated'] and len(data['entries']) == 200
    assert daemon.call('/ui/knowledge/nodes?store=knowledge.db&offset=abc')[0] == 400
    assert daemon.call('/ui/knowledge/stores?path=../private')[0] == 400
    assert daemon.call('/ui/knowledge/stores', method='POST', body={}, ui_header=False)[0] == 403


def test_kind_identity_bounds_refuse_oversized_values(daemon):
    assert daemon.call('/ui/knowledge/nodes?' + urlencode({
        'store': 'knowledge.db', 'kind': 'k' * 201}))[0] == 400
    with GraphStore(daemon.files / 'knowledge.db') as graph:
        graph.upsert_node({'id': 'long-kind', 'kind': 'k' * 40_000, 'label': 'Oversized kind'})
    for route in ('/ui/knowledge/nodes?store=knowledge.db',
                  '/ui/knowledge/node?store=knowledge.db&id=long-kind'):
        code, data, _ = daemon.call(route)
        assert code == 400, data
        assert data == {'error': 'Stored node kinds longer than 200 characters cannot be inspected.'}
    code, data, _ = daemon.call('/ui/knowledge/nodes?store=knowledge.db&kind=model')
    assert code == 400, data
