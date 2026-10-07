"""Training and Settings controls exercised through a real browser."""

import pytest
from test_fleet_ui import Serving

pytestmark = pytest.mark.slow


@pytest.fixture
def usability_page(tmp_path, playwright):
    served = Serving(tmp_path)
    served.ui.settings.setup_done = True
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1280, 'height': 900})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    try:
        yield served, page, errors
    finally:
        browser.close()
        served.close()


def test_training_environment_catalogue_drafts_and_pending_submission(usability_page):
    import json

    from playwright.sync_api import expect
    served, page, errors = usability_page
    page.route('**/ui/gym/catalogue', lambda route: route.fulfill(json={
        'environments': [{'id': 'drone', 'title': 'Forest search drones', 'available': True, 'library': 'PyFlyt'},
                         {'id': 'car', 'title': 'Smart car', 'available': False,
                          'library': 'MetaDrive', 'missing': ['metadrive']}]}))
    page.goto(f'http://127.0.0.1:{served.port}/ui/#training')
    workflow = page.get_by_label('Workflow', exact=True)
    expect(page.get_by_role('button', name='Queue configuration check', exact=True)).to_be_enabled()
    workflow.select_option('rl')
    environment = page.locator('training-view').get_by_label('Environment', exact=True)
    expect(environment).to_have_value('drone')
    page.get_by_label('Training timesteps', exact=True).fill('123')
    environment.select_option('car')
    expect(page.get_by_role('button', name='Start training', exact=True)).to_be_disabled()
    expect(page.locator('training-view .training-form .intro')).to_contain_text('Install MetaDrive')
    environment.select_option('drone')
    expect(page.get_by_label('Training timesteps', exact=True)).to_have_value('123')
    assert not page.get_by_label('Environment configuration (JSON)', exact=True).is_visible()
    page.locator('training-view .training-form details summary').first.click()
    page.get_by_label('Environment configuration (JSON)', exact=True).fill('{"world":{"trees":4}}')
    spec = page.evaluate("document.querySelector('training-view').spec()")
    assert spec['args'][1] == 'drone' and '--dry-run' not in spec['args']
    assert json.loads(spec['args'][spec['args'].index('--config') + 1]) == {'world': {'trees': 4}}
    page.evaluate("""() => {
      const original = window.fleetModel.api;
      window.fleetModel.api = (path, options) => path === '/ui/workspace/jobs' && options?.method === 'POST'
        ? new Promise(resolve => { window.finishTrainingRequest = resolve; }) : original(path, options);
    }""")
    page.get_by_role('button', name='Start training', exact=True).click()
    expect(page.get_by_role('button', name='Working…', exact=True)).to_be_disabled()
    expect(page.get_by_role('button', name='Review command', exact=True)).to_be_disabled()
    page.evaluate("window.finishTrainingRequest({ok:false,status:400,error:'Training capacity unavailable'})")
    expect(page.locator('training-view .training-form .status')).to_contain_text('Training capacity unavailable')
    expect(page.get_by_role('button', name='Start training', exact=True)).to_be_enabled()
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path='/private/tmp/ml-stack-training-audit-mobile.png', full_page=True)
    assert not errors


def test_settings_failure_is_visible_and_only_selected_libraries_install(usability_page):
    from playwright.sync_api import expect
    served, page, errors = usability_page
    payload = {'ready': False, 'libraries': [
        {'name': 'core', 'title': 'Training essentials', 'blurb': 'Common training libraries',
         'default': True, 'installed': False, 'size_mb': 40},
        {'name': 'gym-drone', 'title': 'Forest search drones', 'blurb': 'PyFlyt drone simulator',
         'default': False, 'installed': False, 'size_mb': 400, 'python_version': '3.12'}]}
    posts = []

    def libraries(route):
        if route.request.method == 'POST':
            posts.append(route.request.post_data_json)
            route.fulfill(status=400, json={'error': 'Could not download the selected library'})
        else:
            route.fulfill(json=payload)

    page.route('**/ui/libraries', libraries)
    page.route('**/ui/settings', lambda route: route.fulfill(status=400, json={'error': 'Preferences are read-only'})
               if route.request.method == 'POST' else route.continue_())
    page.goto(f'http://127.0.0.1:{served.port}/ui/#settings')
    expect(page.locator('#settings-save')).to_be_enabled()
    page.locator('#settings-save').click()
    expect(page.locator('#settings-note .err')).to_contain_text('Preferences are read-only')
    expect(page.locator('#settings-save')).to_be_enabled()
    expect(page.locator('[data-pane="libraries"]')).not_to_be_visible()
    page.get_by_role('tab', name='Libraries & simulators', exact=True).click()
    expect(page.locator('#lib-core')).not_to_be_checked()
    page.locator('#lib-gym-drone').check()
    page.get_by_role('button', name='Apply library changes', exact=True).click()
    expect(page.locator('#settings-libs .err')).to_contain_text('Could not download the selected library')
    assert posts == [{'install': ['gym-drone'], 'remove': []}]
    expect(page.get_by_role('button', name='Apply library changes', exact=True)).to_be_enabled()
    page.screenshot(path='/private/tmp/ml-stack-settings-audit.png', full_page=True)
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors


def test_navigation_search_shortcuts_no_results_and_mobile_width(usability_page):
    from playwright.sync_api import expect
    served, page, errors = usability_page
    page.goto(f'http://127.0.0.1:{served.port}/ui/#chat')
    page.keyboard.press('Control+k')
    search = page.get_by_role('searchbox', name='Find a page')
    expect(search).to_be_focused()
    search.fill('not-a-workspace-page')
    expect(page.locator('#nav-empty')).to_be_visible()
    search.press('Escape')
    expect(page.locator('#nav-empty')).not_to_be_visible()
    search.fill('training')
    search.press('Enter')
    expect(page.locator('#nav-title')).to_have_text('Training')
    expect(page.locator('training-view section')).to_be_visible()
    search.fill('board')
    page.evaluate("window.fleetModel.go('settings')")
    expect(page.locator('#nav-title')).to_have_text('Settings')
    search.press('Escape')
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert page.locator('fleet-nav a[href="#board"]').count() == 1
    assert not errors


def test_settings_queued_install_keeps_navigation_usable_until_ready(usability_page):
    from playwright.sync_api import expect
    served, page, errors = usability_page
    installed = False
    job = {'id': 'setup-core', 'state': 'queued', 'note': 'Waiting to install', 'result': {}}
    def libraries(route):
        if route.request.method == 'POST':
            route.fulfill(status=202, json={'ok': True, 'job': job})
        else:
            route.fulfill(json={'libraries': [{'name': 'core', 'title': 'Training essentials', 'blurb': 'Training tools', 'installed': installed, 'size_mb': 40}]})
    page.route('**/ui/libraries', libraries)
    page.route('**/ui/setup/jobs', lambda route: route.fulfill(json={'jobs': [job]}))
    page.goto(f'http://127.0.0.1:{served.port}/ui/#settings')
    page.get_by_role('tab', name='Libraries & simulators', exact=True).click()
    page.locator('#lib-core').check()
    apply = page.get_by_role('button', name='Apply library changes', exact=True)
    apply.click()
    expect(apply).to_be_enabled()
    expect(page.locator('#settings-libs')).to_contain_text('Waiting to install')
    page.evaluate("location.hash = '#chat'")
    expect(page.locator('chat-view')).to_be_visible()
    job.update(state='installing', note='Downloading training tools')
    page.evaluate("location.hash = '#settings'")
    page.get_by_role('tab', name='Libraries & simulators', exact=True).click()
    expect(page.locator('#lib-core')).not_to_be_checked()
    installed = True
    job.update(state='done', result={'changed': {'core': {'ok': True}}})
    expect(page.locator('#lib-core')).to_be_checked(timeout=5000)
    expect(page.locator('#settings-libs')).to_contain_text('Library changes applied')
    assert not errors
