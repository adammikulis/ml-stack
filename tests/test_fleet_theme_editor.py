"""User interaction with saved and previewed appearance."""

import pytest
import test_fleet_page as fleet_page

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
no_release_lookup = fleet_page.no_release_lookup
open_page = fleet_page.open_page

pytestmark = pytest.mark.slow


def mount_editor(page):
    page.get_by_role('tab', name='Appearance & advanced', exact=True).click()
    assert page.locator('theme-editor').count() == 1
    page.get_by_role('button', name='Poolside Light', exact=True).wait_for()


def test_theme_selector_preview_save_reload_reset_and_delete(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#settings')
    mount_editor(page)
    page.get_by_role('button', name='Poolside Light', exact=True).click()
    page.wait_for_function("document.querySelector('theme-editor select').value === 'light'")
    page.locator('theme-editor summary').click()
    page.get_by_label('Pink accent', exact=True).fill('#aa3377')
    page.get_by_label('Interface font size').fill('17')
    page.get_by_label('Interface density').select_option('compact')
    page.get_by_label('Custom theme name').fill('Quiet water')
    assert page.evaluate("document.documentElement.dataset.theme") == 'light'
    assert page.evaluate("document.documentElement.dataset.density") == 'compact'
    assert page.evaluate("document.documentElement.style.getPropertyValue('--ui-font-size')") == '17px'
    assert page.evaluate("document.documentElement.style.getPropertyValue('--poolside-pink')") == '#aa3377'
    _, current, _ = joined.call('/ui/settings', cookie=joined.cookie)
    assert current['settings']['appearance']['active_theme'] == 'light'
    page.get_by_role('button', name='Save theme', exact=True).click()
    page.wait_for_function("document.querySelector('theme-editor select').value === 'custom-1'")
    _, saved, _ = joined.call('/ui/settings', cookie=joined.cookie)
    assert saved['resolved_theme']['colors']['pink'] == '#aa3377'
    assert saved['resolved_theme']['font_size'] == 17
    assert saved['resolved_theme']['density'] == 'compact'
    page.reload()
    mount_editor(page)
    assert page.get_by_label('Saved theme').input_value() == 'custom-1'
    assert page.get_by_label('Pink accent', exact=True).input_value() == '#aa3377'
    page.get_by_label('Pink accent', exact=True).fill('url(x)')
    page.get_by_role('button', name='Save theme', exact=True).click()
    assert page.get_by_label('Pink accent', exact=True).get_attribute('aria-invalid') == 'true'
    _, unchanged, _ = joined.call('/ui/settings', cookie=joined.cookie)
    assert unchanged['settings']['appearance'] == saved['settings']['appearance']
    page.get_by_role('button', name='Reset to preset', exact=True).click()
    page.wait_for_function("document.querySelector('theme-editor select').value === 'light'")
    assert page.get_by_label('Pink accent', exact=True).input_value() == '#ff5fa2'
    page.get_by_label('Saved theme').select_option('custom-1')
    page.get_by_role('button', name='Delete custom theme', exact=True).wait_for()
    page.get_by_role('button', name='Delete custom theme', exact=True).click()
    page.wait_for_function("document.querySelector('theme-editor select').value === 'light'")
    _, deleted, _ = joined.call('/ui/settings', cookie=joined.cookie)
    assert deleted['settings']['appearance']['saved_themes'] == []
    assert not errors
