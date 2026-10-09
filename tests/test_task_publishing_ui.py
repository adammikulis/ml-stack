"""The assembled Tasks page exposes publishing evidence without offering new authority."""

import pytest
from browser_expect import expect
from test_fleet_ui import Serving

pytestmark = pytest.mark.slow


def test_mobile_publishing_details_escape_evidence_and_survive_refresh(tmp_path, playwright):
    server = Serving(tmp_path)
    server.ui.settings.setup_done = True
    project = '/private/tmp/' + 'very-long-approved-worktree-directory/' * 8 + 'reviewed-task'
    proof = {'id': 'integration:demo', 'state': 'blocked', 'owner': 'coordinator',
             'review_id': 'review:independent', 'review_hash': 'a' * 64,
             'proposal_id': 'proposal:replay', 'proposal_hash': 'b' * 64,
             'source_commit': 'c' * 40, 'commit': 'd' * 40, 'development': '0.2dev',
             'reason': '<img src=x> Foreign worktree remains active', 'blocking_owner': 'other-device',
             'blocking_claim': {'kind': 'branch', 'owner': 'other-device', 'key': '0.2dev'},
             'checks': [{'command': ['scripts/test', 'gate'], 'passed': True, 'exit': 0, 'output_hash': 'e' * 64}]}
    task = {'id': 'task:' + '1' * 32, 'title': 'Reviewed replay', 'description': 'Accepted source awaiting publishing.',
            'state': 'completed', 'acceptance': ['Replay passes'], 'worker': 'device-worker',
            'lease': {'resource': {'project': project}}, 'integration': proof, 'integrations': [proof],
            'integration_attempts': [{'id': 'integration-attempt:once', 'state': 'blocked',
                                     'started': 1791129600, 'finished': 1791129615,
                                     'review_id': proof['review_id'], 'review_hash': proof['review_hash'],
                                     'proposal_id': proof['proposal_id'], 'proposal_hash': proof['proposal_hash'],
                                     'outcome': proof}],
            'integration_events': [{'state': 'candidate', 'at': 1791129601}, {'state': 'blocked', 'at': 1791129615}]}
    posts = []
    def respond(route):
        if route.request.method == 'POST':
            posts.append(route.request.post_data_json)
        route.fulfill(json={'tasks': [task], 'metrics': {}})
    try:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page(viewport={'width': 390, 'height': 844})
            page.route('**/ui/tasks', respond)
            page.goto(f'http://127.0.0.1:{server.port}/ui/#tasks')
            viewer = page.locator('tasks-view')
            viewer.get_by_role('button', name='Reviewed replay').click()
            project_view = viewer.locator('.task-project')
            expect(project_view.locator('summary')).to_have_text('reviewed-task')
            expect(project_view.locator('code')).not_to_be_visible()
            publishing = viewer.locator('.task-publishing')
            expect(publishing.locator('.publishing-reason').first).not_to_be_visible()
            page.screenshot(path='/private/tmp/ml-stack-task-publishing-collapsed-mobile.png', full_page=True)
            publishing.locator(':scope > summary').focus()
            page.keyboard.press('Enter')
            expect(publishing.get_by_text('other-device', exact=True).first).to_be_visible()
            expect(publishing.get_by_text('Independent review', exact=True).first).to_be_visible()
            expect(publishing.get_by_text('proposal:replay', exact=True).first).to_be_visible()
            gate = publishing.locator('.publishing-gate').first
            gate.locator('summary').click()
            expect(gate.get_by_text('0', exact=True)).to_be_visible()
            expect(gate.get_by_text('e' * 64, exact=True)).to_be_visible()
            attempt = publishing.locator('[data-publishing-key="integration-attempt:once"]')
            attempt.locator(':scope > summary').click()
            expect(attempt.get_by_text('Started', exact=True)).to_be_visible()
            expect(attempt.get_by_text('Finished', exact=True)).to_be_visible()
            project_view.locator('summary').click()
            expect(project_view.locator('code')).to_have_text(project)
            proof['reason'] = 'Owner is resolving the publication block'
            viewer.get_by_role('button', name='Refresh', exact=True).click()
            expect(publishing.locator('.publishing-reason').first).to_have_text(proof['reason'])
            expect(project_view.locator('code')).to_be_visible()
            expect(publishing.locator('.publishing-gate').first.get_by_text('e' * 64, exact=True)).to_be_visible()
            assert viewer.locator('img').count() == 0 and publishing.get_by_role('button').count() == 0
            assert posts == []
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path='/private/tmp/ml-stack-task-publishing-mobile.png', full_page=True)
    finally:
        server.close()
