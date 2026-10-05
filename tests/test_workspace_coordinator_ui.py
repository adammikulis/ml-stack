"""The person can inspect shared routing without confusing it with a local Board."""

import pytest
from playwright.sync_api import expect
from test_fleet_ui import Serving

pytestmark = pytest.mark.slow


def test_mobile_shared_coordinator_status_hides_local_board_and_escapes_labels(tmp_path, playwright):
    server = Serving(tmp_path)
    server.ui.settings.setup_done = True
    try:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page(viewport={'width': 390, 'height': 844})
            page.route('**/ui/coordination', lambda route: route.fulfill(json={
                'mode': 'remote', 'connected': False, 'name': '<img src=x> Windows coordinator',
                'endpoint': 'https://coordinator.example:8770', 'workspace': 'workspace:' + 'a' * 32}))
            page.goto(f'http://127.0.0.1:{server.port}/ui/#board')
            view = page.locator('coordinator-control')
            expect(view).to_contain_text('Unavailable; local fallback is disabled')
            expect(view).to_contain_text('<img src=x> Windows coordinator')
            assert view.locator('img').count() == 0
            expect(view.get_by_role('link', name='Open coordinator Board')).to_have_attribute(
                'href', 'https://coordinator.example:8770/ui/#board')
            expect(page.locator('#board-agents')).to_be_hidden()
            assert page.locator('board-view ml-board').count() == 0
            expect(view.locator('details')).not_to_have_attribute('open', '')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    finally:
        server.close()
