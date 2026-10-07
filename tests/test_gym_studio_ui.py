"""Scenario selection and studio layout in the browser."""

import pytest
import test_fleet_page as fleet_page

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
no_release_lookup = fleet_page.no_release_lookup
open_page = fleet_page.open_page
pytestmark = pytest.mark.slow


def test_scenario_cards_drive_setup_and_preserve_live_control_access(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#gym')
    page.wait_for_function("document.querySelector('gym-view')?.catalogue.length > 0")
    page.locator('gym-view .gym-scenario[data-environment="warehouse"] button').first.click()
    assert page.get_by_label('Environment', exact=True).input_value() == 'warehouse'
    assert page.get_by_label('Controller', exact=True).input_value() == 'manual'
    assert page.locator('gym-view .gym-scenario[data-environment="warehouse"] button').first.get_attribute('aria-pressed') == 'true'
    page.locator('gym-view .gym-scenario[data-environment="car"] button').first.click()
    assert page.get_by_label('Environment', exact=True).input_value() == 'car'
    assert page.get_by_label('Controller', exact=True).input_value() == 'decider'
    assert not page.locator('gym-view #config .gym-advanced').get_attribute('open')
    assert not page.get_by_label('Seed', exact=True).is_visible()
    assert not page.locator('gym-view .gym-world-settings').get_attribute('open')
    start = page.get_by_role('button', name='Start session', exact=True).bounding_box()
    assert start and start['y'] + start['height'] < page.viewport_size['height']
    assert page.locator('gym-view .gym-stage').is_visible()
    assert page.get_by_role('button', name='Single step', exact=True).is_visible()
    assert not page.get_by_role('button', name='Apply manual action', exact=True).is_visible()
    assert not page.get_by_role('button', name='Refresh recordings', exact=True).is_visible()
    page.locator('gym-view .gym-recordings-panel > summary').click()
    assert page.get_by_role('button', name='Refresh recordings', exact=True).is_visible()
    page.locator('gym-view .gym-recordings-panel > summary').click()
    layout = page.evaluate("""() => {
      const g=document.querySelector('gym-view'),setup=g.querySelector('#config').getBoundingClientRect(),
        stage=g.querySelector('.gym-stage').getBoundingClientRect();
      return {setup:setup.width,stage:stage.width,canvasRight:stage.left>=setup.right};
    }""")
    assert layout['canvasRight'] and layout['stage'] > layout['setup']
    page.locator('gym-view #config .gym-advanced > summary').click()
    assert page.get_by_label('Seed', exact=True).is_visible()
    assert not errors
