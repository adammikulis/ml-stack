"""Tasks UI makes verified outcomes and independent evidence distinct from worker progress."""

import pytest
from playwright.sync_api import expect
from test_fleet_ui import Serving

from ml_stack.workspace import task_summary

pytestmark = pytest.mark.slow


def test_tasks_filters_artifacts_and_independent_review_payload(tmp_path, playwright):
    server = Serving(tmp_path)
    server.ui.settings.setup_done = True
    task = {'id': 'task:' + 'a' * 32, 'title': 'Native road replay', 'description': '<img src=x> inspect sensors',
            'state': 'review', 'acceptance': ['Replay passes'], 'worker': 'worker-a', 'base_id': 'device-account',
            'device_id': 'device-1', 'lease': {'resource': {'model': 'Strands2B', 'project': '/approved/worktree'}},
            'created_at': 1, 'checkpoints': [{'at': 2, 'summary': 'Worker claims progress'}],
            'proposal': {'artifacts': {'replay.json': 'a' * 64}}, 'reviews': []}
    posts = []
    def respond(route):
        if route.request.method == 'POST':
            posts.append(route.request.post_data_json)
            task['state'] = 'completed' if posts[-1]['action'] == 'review' else 'queued'
            route.fulfill(json={'accepted': True})
        else:
            task['activity'] = task_summary.inspection(task, 100)
            route.fulfill(json={'tasks': [task], 'overview': task_summary.overview([task], 100), 'metrics': {'verified_outcomes': int(task['state'] == 'completed'),
                                                          'accepted_artifacts': int(task['state'] == 'completed'),
                                                          'failures': 0, 'blocked_seconds': 0}})
    try:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page(viewport={'width': 390, 'height': 844})
            project_id = 'b' * 32
            page.route('**/ui/projects', lambda route: route.fulfill(json={'workspaces': [{'id': project_id, 'name': 'Demo workspace', 'local_authority': True, 'board_host': 'http://canonical', 'authority_machine': 'device-1'}]}))
            page.route(f'**/ui/projects/{project_id}/tasks', respond)
            page.goto(f'http://127.0.0.1:{server.port}/ui/#tasks')
            viewer = page.locator('tasks-view')
            expect(viewer.locator('.task-remaining-number')).to_have_text('1')
            expect(viewer.locator('[data-task-count=review] strong')).to_have_text('1')
            expect(viewer.locator('.task-eta-value')).to_have_text('Unknown')
            viewer.get_by_role('button', name='Native road replay').click()
            expect(viewer.get_by_text('Strands2B', exact=True)).to_be_visible()
            viewer.get_by_label('Task status').select_option('blocked')
            expect(viewer.get_by_text('No matching tasks.', exact=True)).to_be_visible()
            viewer.get_by_label('Task status').select_option('review')
            viewer.get_by_text('Submitted proposal and artifact hashes', exact=True).click()
            expect(viewer.locator('pre').filter(has_text='replay.json')).to_be_visible()
            assert viewer.locator('img').count() == 0
            viewer.get_by_text('Independently review this proposal', exact=True).click()
            viewer.get_by_label('Independent review reason').fill('Replay independently verified')
            viewer.get_by_label('Replay passes', exact=True).check()
            viewer.get_by_role('button', name='Record independent review').click()
            expect(viewer.locator('.badge')).to_have_text('completed')
            assert posts == [{'action': 'review', 'id': task['id'], 'decision': {'accepted': True,
                             'outcome': 'accepted', 'reason': 'Replay independently verified',
                             'checks': [{'name': 'Replay passes', 'passed': True}]}}]
            task.update(state='blocked', blocked_reason='Approval pending')
            viewer.get_by_label('Task status').select_option('')
            viewer.get_by_role('button', name='Refresh', exact=True).click()
            viewer.get_by_label('Continuation reason').fill('Person resolved approval')
            viewer.get_by_role('button', name='Authorize resume').click()
            expect(viewer.locator('.badge')).to_have_text('queued')
            assert posts[-1] == {'action': 'resume', 'id': task['id'], 'reason': 'Person resolved approval'}
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path='/private/tmp/ml-stack-tasks-mobile.png', full_page=True)
    finally:
        server.close()


@pytest.mark.parametrize('status', [403, 503])
def test_unavailable_tasks_hide_unknown_lanes_and_recover(tmp_path, playwright, status):
    server = Serving(tmp_path)
    server.ui.settings.setup_done = True
    project_id = 'b' * 32
    failing = status == 403
    task = {'id': 'task:' + 'a' * 32, 'title': 'Previously loaded task', 'description': '',
            'state': 'queued', 'acceptance': ['Verified outcome'], 'created_at': 1}

    def respond(route):
        if failing:
            route.fulfill(status=status, json={'error': 'Project access unavailable'})
        else:
            route.fulfill(json={'tasks': [task], 'overview': task_summary.overview([task], 100)})

    try:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page()
            page.route('**/ui/projects', lambda route: route.fulfill(json={'workspaces': [
                {'id': project_id, 'name': 'Test workspace', 'local_authority': True, 'board_host': 'http://canonical'}]}))
            page.route(f'**/ui/projects/{project_id}/tasks', respond)
            page.goto(f'http://127.0.0.1:{server.port}/ui/#tasks')
            viewer = page.locator('tasks-view')
            if not failing:
                expect(viewer.locator('.task-row').filter(has_text='Previously loaded task')).to_be_visible()
                failing = True
                viewer.get_by_role('button', name='Refresh', exact=True).click()
            expect(viewer.locator('.status')).to_contain_text('Task data unavailable:')
            expect(viewer.locator('.task-layout')).to_be_hidden()
            expect(viewer.locator('.task-column')).to_have_count(0)
            expect(viewer.get_by_text('No matching tasks.', exact=True)).to_have_count(0)
            expect(viewer.locator('.task-remaining-number')).to_have_text('—')
            expect(viewer.locator('[data-task-count=queued] strong')).to_have_text('—')
            expect(viewer.get_by_role('button', name='Open Projects', exact=True)).to_be_visible()
            page.screenshot(path=f'/private/tmp/poolside-tasks-unavailable-{status}.png', full_page=True)
            failing = False
            viewer.get_by_role('button', name='Refresh', exact=True).click()
            expect(viewer.locator('.task-row').filter(has_text='Previously loaded task')).to_be_visible()
            expect(viewer.locator('.task-remaining-number')).to_have_text('1')
            expect(viewer.get_by_label('Search tasks')).to_be_enabled()
    finally:
        server.close()


def test_failed_previous_project_does_not_replace_current_project_status(tmp_path, playwright):
    server = Serving(tmp_path)
    server.ui.settings.setup_done = True
    first, second = 'b' * 32, 'c' * 32
    pending = []
    try:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page()
            page.route('**/ui/projects', lambda route: route.fulfill(json={'workspaces': [
                {'id': key, 'name': name, 'local_authority': True, 'board_host': 'http://canonical'}
                for key, name in [(first, 'Previous project'), (second, 'Current project')]]}))
            page.route(f'**/ui/projects/{first}/tasks', lambda route: pending.append(route))
            page.route(f'**/ui/projects/{second}/tasks', lambda route: route.fulfill(json={
                'tasks': [], 'overview': task_summary.overview([], 100)}))
            page.goto(f'http://127.0.0.1:{server.port}/ui/#tasks')
            viewer = page.locator('tasks-view')
            expect(viewer.locator('.status')).to_have_text('Updating task overview…')
            page.wait_for_function("document.querySelector('tasks-view').loading")
            viewer.get_by_label('Task project workspace').select_option(second)
            page.wait_for_timeout(50)
            assert len(pending) == 1
            page.evaluate("""() => {
                window.taskStatusHistory = [];
                const note = document.querySelector('tasks-view .status');
                new MutationObserver(records => {
                    for (const record of records) for (const node of record.addedNodes)
                        window.taskStatusHistory.push(node.textContent);
                })
                    .observe(note, {childList:true, subtree:true, characterData:true});
            }""")
            pending[0].fulfill(status=403, json={'error': 'Previous project denied'})
            expect(viewer.locator('.status')).to_contain_text('Updated ')
            expect(viewer.locator('.status')).not_to_contain_text('Previous project denied')
            expect(viewer.locator('.task-remaining-number')).to_have_text('0')
            expect(viewer.locator('.task-layout')).to_be_visible()
            assert all('Task data unavailable:' not in text for text in page.evaluate('taskStatusHistory'))
    finally:
        server.close()


@pytest.mark.parametrize('foreign_authority', [False, True])
def test_tasks_join_selected_person_workspace_with_real_backend(tmp_path, playwright, monkeypatch, foreign_authority):
    from ml_stack.fleet import project_client
    from ml_stack.fleet.projects import ProjectRegistry, identity
    from ml_stack.net import git
    from ml_stack.workspace.remote_host import WorkspaceHost

    checkout = tmp_path / 'experiment'
    checkout.mkdir()
    git.run(['init'], cwd=checkout)
    git.run(['remote', 'add', 'origin', 'https://code.example.invalid/team/tasks.git'], cwd=checkout)
    registry = ProjectRegistry(tmp_path / 'registry', 'fixture-device', (checkout,), 'http://127.0.0.1:8770')
    project = identity(checkout)
    host = WorkspaceHost(registry)
    host.prepare(project)
    ws = host.workspace(project)
    monkeypatch.setattr(project_client, 'peers', lambda ui: [])
    server = Serving(tmp_path / 'ui')
    server.ui.settings.setup_done = True
    server.ui.projects, server.ui.workspaces = registry, host
    session = server.ui.sessions.open('task-person')
    posts = []
    try:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page()
            page.context.add_cookies([{'name': 'ml_stack_ui', 'value': session.sid,
                                      'url': f'http://127.0.0.1:{server.port}/ui', 'httpOnly': True}])
            page.on('request', lambda request: posts.append(request.url) if request.method == 'POST' and request.url.endswith('/board/connect') else None)
            page.goto(f'http://127.0.0.1:{server.port}/ui/#tasks')
            viewer = page.locator('tasks-view')
            expect(viewer.get_by_label('Task project workspace')).to_have_value(project)
            join = viewer.get_by_role('button', name='Join workspace as person', exact=True)
            expect(join).to_be_visible()
            assert not posts
            assert all('person_project' not in row for row in ws.registry._load().values())
            if foreign_authority:
                registry.get(project).authority_machine = 'foreign-device'
            join.click()
            assert posts == [f'http://127.0.0.1:{server.port}/ui/projects/{project}/board/connect']
            if foreign_authority:
                expect(viewer.locator('.status')).to_contain_text('authoritative device')
                expect(viewer.locator('.task-layout')).to_be_hidden()
                assert all('person_project' not in row for row in ws.registry._load().values())
            else:
                expect(viewer.locator('.task-layout')).to_be_visible()
                expect(viewer.locator('.task-remaining-number')).to_have_text('0')
                expect(join).to_have_count(0)
            page.screenshot(path=f'/private/tmp/poolside-tasks-person-join-{foreign_authority}.png', full_page=True)
    finally:
        server.close()


def test_task_join_completion_does_not_reload_another_selected_project(tmp_path, playwright):
    server = Serving(tmp_path)
    server.ui.settings.setup_done = True
    first, second = 'b' * 32, 'c' * 32
    pending, current_reads = [], []

    def current(route):
        current_reads.append(route.request.url)
        route.fulfill(json={'tasks': [], 'overview': task_summary.overview([], 100)})

    try:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page()
            page.route('**/ui/projects', lambda route: route.fulfill(json={'workspaces': [
                {'id': key, 'name': name, 'local_authority': True, 'board_host': 'http://canonical'}
                for key, name in [(first, 'Join this project'), (second, 'Other project')]]}))
            page.route(f'**/ui/projects/{first}/tasks', lambda route: route.fulfill(status=503, json={
                'error': 'Join this workspace as person to use conversations and tasks.', 'person_setup_required': True}))
            page.route(f'**/ui/projects/{first}/board/connect', lambda route: pending.append(route))
            page.route(f'**/ui/projects/{second}/tasks', current)
            page.goto(f'http://127.0.0.1:{server.port}/ui/#tasks')
            viewer = page.locator('tasks-view')
            join = viewer.get_by_role('button', name='Join workspace as person', exact=True)
            expect(join).to_be_visible()
            page.evaluate("window.taskJoinButton = [...document.querySelectorAll('tasks-view button')].find(node=>node.textContent==='Join workspace as person')")
            join.click()
            expect(join).to_be_disabled()
            viewer.get_by_label('Task project workspace').select_option(second)
            expect(viewer.locator('.status')).to_contain_text('Updated ')
            assert len(pending) == len(current_reads) == 1
            pending[0].fulfill(json={'me': 'person', 'project_id': first})
            page.wait_for_function('window.taskJoinButton.disabled === false')
            assert len(current_reads) == 1
            expect(viewer.get_by_label('Task project workspace')).to_have_value(second)
            expect(viewer.locator('.status')).to_contain_text('Updated ')
    finally:
        server.close()
