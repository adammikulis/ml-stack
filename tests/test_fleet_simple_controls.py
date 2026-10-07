"""Accurate labels and expandable expert controls through the browser."""

import pytest
import test_fleet_page as fleet_page

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
no_release_lookup = fleet_page.no_release_lookup
open_page = fleet_page.open_page

pytestmark = pytest.mark.slow


def test_context_length_uses_tokens_and_saves_the_chosen_limit(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#settings')
    page.get_by_role('tab', name='Models & providers').click()
    slider = page.get_by_role('slider', name='Context length')
    slider.wait_for()
    assert slider.get_attribute('aria-valuetext') == '8,192 tokens'
    slider.focus()
    slider.press('ArrowRight')
    assert slider.get_attribute('aria-valuetext') == '16,384 tokens'
    page.click('#settings-save')
    page.wait_for_selector('#settings-note .ok')
    _, saved, _ = joined.call('/ui/settings',cookie=joined.cookie)
    assert saved['settings']['context'] == 16384
    assert not page.locator('#settings-removal button.danger').is_visible()
    page.get_by_role('tab', name='Maintenance', exact=True).click()
    page.locator('#settings-removal button.danger').wait_for()
    page.get_by_role('tab', name='Models & providers').click()
    slider.focus()
    slider.press('End')
    assert slider.get_attribute('aria-valuetext') == '1,048,576 tokens'
    assert page.get_by_text('Extended context: Qwen uses YaRN above its native 256K window.', exact=False).is_visible()
    page.click('#settings-save')
    page.wait_for_selector('#settings-note .ok')
    _, saved, _ = joined.call('/ui/settings', cookie=joined.cookie)
    assert saved['settings']['context'] == 1048576
    assert page.evaluate('''() => {
      const el = window.fleetModel.el;
      return !el('option', {disabled: false}).disabled
        && el('option', {disabled: true}).disabled
        && el('div', {'aria-expanded': false}).getAttribute('aria-expanded') === 'false';
    }''')
    assert not errors


def test_training_keeps_required_fields_visible_and_advanced_values_are_used(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie,path='/ui/#training')
    page.get_by_label('Workflow',exact=True).select_option('decider')
    dataset = page.get_by_label('Dataset path (relative to files root)',exact=True)
    dataset.wait_for()
    assert page.get_by_label('Training steps',exact=True).is_visible()
    learning_rate = page.get_by_label('Learning rate',exact=True)
    assert not learning_rate.is_visible()
    page.locator('training-view #config details').filter(has=page.get_by_text('Advanced options',exact=True)).locator('summary').click()
    learning_rate.fill('0.0003')
    dataset.fill('datasets/actions.jsonl')
    page.get_by_role('button',name='Review command',exact=True).click()
    preview = page.locator('training-view #config pre')
    preview.wait_for()
    assert '--lr 0.0003' in preview.inner_text()
    assert '--data datasets/actions.jsonl' in preview.inner_text()
    page.get_by_label('Workflow',exact=True).select_option('rl')
    assert page.locator('training-view').get_by_label('Environment',exact=True).is_visible()
    assert not dataset.is_visible()
    assert page.get_by_label('Training timesteps',exact=True).is_visible()
    assert not errors


def test_decision_lab_shows_only_inputs_for_the_selected_operation(joined, open_page):
    page, errors = open_page(joined,cookie=joined.cookie,path='/ui/#tools')
    assert page.get_by_label('Question',exact=True).is_visible()
    assert not page.get_by_label('Evaluation cases (JSONL path or guards)',exact=True).is_visible()
    assert not page.get_by_label('Decision backend',exact=True).is_visible()
    page.get_by_label('Operation',exact=True).select_option('eval')
    assert page.get_by_label('Evaluation cases (JSONL path or guards)',exact=True).is_visible()
    assert not page.get_by_label('Question',exact=True).is_visible()
    page.locator('tools-view #decision details summary').click()
    assert page.get_by_label('Decision backend',exact=True).is_visible()
    assert not errors


def test_interactive_chat_command_is_given_to_the_person_without_a_job(joined, open_page):
    page, errors = open_page(joined,cookie=joined.cookie,path='/ui/#tools')
    page.get_by_role('tab', name='Command library', exact=True).click()
    runner = page.locator('tools-view #runner')
    command = runner.get_by_label('Installed command',exact=True)
    command.select_option('ml-stack-chat')
    runner.locator('details summary').click()
    runner.get_by_label('Arguments as JSON array',exact=True).fill('[]')
    runner.get_by_role('button',name='Run command',exact=True).click()
    page.wait_for_function("document.querySelector('tools-view #runner pre').textContent.includes('Run this command in your own terminal')")
    assert 'ml-stack-chat' in runner.locator('pre').inner_text()
    assert joined.runner.snapshot() == []
    assert not errors


def test_apple_memory_is_one_physical_pool(joined, open_page):
    joined.ui.peers = lambda force=False: [{
        'name': 'local-mac', 'is_self': True, 'slots': 1, 'free': 1,
        'device': {'vendor': 'apple', 'unified_memory': True,
                   'ram_gb': 128, 'ram_used_gb': 45,
                   'vram_total_gb': 112, 'vram_free_gb': 112,
                   'serving': [{'models': ['Qwen3.8-27B'], 'context': 262144,
                                'slots': 1, 'spec_type': 'draft-mtp', 'draft_status': 'active'}]}}]
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#cluster')
    page.get_by_text('45.0 GB of 128.0 GB in use', exact=False).wait_for()
    meters = page.locator('cluster-view ml-meter')
    assert meters.filter(has=page.get_by_text('memory', exact=True)).count() == 1
    assert page.get_by_text('unified memory', exact=True).count() == 0
    assert page.locator('.loaded-models').get_by_text('Qwen3.8-27B', exact=True).is_visible()
    assert page.locator('.loaded-models').get_by_text('262,144 tokens · MTP active · 1 slot(s)', exact=True).is_visible()
    assert page.get_by_text('112.0 GB free of 112.0 GB', exact=False).count() == 0
    assert not errors


def test_central_advanced_preference_expands_pages_and_survives_reload(joined, open_page):
    from playwright.sync_api import expect

    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#settings')
    preference = page.get_by_label('Always show advanced options', exact=True)
    expect(preference).not_to_be_checked()
    preference.check()
    page.click('#settings-save')
    expect(page.locator('#settings-note .ok')).to_contain_text('Preferences saved')
    _, saved, _ = joined.call('/ui/settings', cookie=joined.cookie)
    assert saved['settings']['always_show_advanced'] is True
    page.evaluate("window.fleetModel.go('training')")
    page.get_by_label('Workflow', exact=True).select_option('decider')
    expect(page.get_by_label('Learning rate', exact=True)).to_be_visible()
    page.reload()
    expect(page.locator('training-view #config details[data-advanced]')).to_have_attribute('open', '')
    page.evaluate("""() => {
      const panel = window.fleetModel.el('details', {}, window.fleetModel.el('summary', {}, 'Advanced options'));
      panel.id = 'dynamic-advanced'; document.body.append(panel);
    }""")
    expect(page.locator('#dynamic-advanced')).to_have_attribute('open', '')
    page.evaluate("window.fleetModel.go('settings')")
    expect(preference).to_be_checked()
    preference.uncheck()
    page.click('#settings-save')
    expect(page.locator('#settings-note .ok')).to_contain_text('Preferences saved')
    assert not errors


def test_sign_in_loads_advanced_preference_from_backend(joined, open_page):
    from playwright.sync_api import expect

    joined.ui.settings.always_show_advanced = True
    joined.ui.settings.save(joined.ui.settings_path)
    page, errors = open_page(joined, path='/ui/#training')
    page.locator('sign-in input[type="password"]').fill(fleet_page.WORDS)
    page.locator('sign-in button').click()
    expect(page.locator('training-view #config details[data-advanced]')).to_have_attribute('open', '')
    assert not errors


def test_settings_backend_validation_error_stays_visible(joined, open_page):
    from playwright.sync_api import expect

    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#settings')
    expect(page.locator('#settings-save')).to_be_enabled()
    page.route('**/ui/settings', lambda route: route.fulfill(json={'error': 'Choose a download source'})
               if route.request.method == 'POST' else route.continue_())
    page.locator('#settings-save').click()
    expect(page.locator('#settings-note .err')).to_contain_text('Choose a download source')
    assert not errors


def test_central_output_default_saves_and_survives_reloading(joined, open_page):
    from playwright.sync_api import expect

    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#settings')
    page.get_by_role('tab', name='Models & providers').click()
    output = page.get_by_label('Default maximum output tokens', exact=True)
    expect(output).to_have_value('8192')
    output.fill('2048')
    page.locator('#settings-save').click()
    expect(page.locator('#settings-note .ok')).to_contain_text('Preferences saved')
    _, defaults, _ = joined.call('/ui/conversations/defaults', cookie=joined.cookie)
    assert defaults['settings']['max_output_tokens'] == 2048
    page.reload()
    page.get_by_role('tab', name='Models & providers').click()
    expect(output).to_have_value('2048')
    output.fill('')
    page.locator('#settings-save').click()
    expect(page.locator('#settings-note .ok')).to_contain_text('Preferences saved')
    _, defaults, _ = joined.call('/ui/conversations/defaults', cookie=joined.cookie)
    assert defaults['settings']['max_output_tokens'] is None
    page.reload()
    page.get_by_role('tab', name='Models & providers').click()
    expect(output).to_have_value('')
    assert not errors


def test_settings_switch_sections_without_losing_unsaved_values(joined, open_page):
    from playwright.sync_api import expect

    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#settings')
    panels = page.locator('settings-view [data-pane]:visible')
    expect(panels).to_have_count(1)
    expect(panels).to_have_attribute('data-pane', 'appearance')
    expect(page.locator('#settings-removal')).not_to_be_visible()
    page.get_by_role('tab', name='Models & providers').click()
    output = page.get_by_label('Default maximum output tokens', exact=True)
    output.fill('4096')
    page.get_by_role('tab', name='Compute & device').click()
    expect(panels).to_have_count(1)
    expect(panels).to_have_attribute('data-pane', 'compute')
    page.get_by_role('tab', name='Models & providers').click()
    expect(output).to_have_value('4096')
    page.get_by_role('button', name='Save preferences', exact=True).click()
    expect(page.locator('#settings-note .ok')).to_contain_text('Preferences saved')
    page.reload()
    page.get_by_role('tab', name='Models & providers').click()
    expect(output).to_have_value('4096')
    page.get_by_role('tab', name='Maintenance', exact=True).click()
    expect(page.locator('#settings-removal button.danger')).to_be_visible()
    expect(page.locator('#settings-chatting')).not_to_be_visible()
    page.screenshot(path='/private/tmp/poolside-rebuild-settings-maintenance.png', full_page=True)
    assert not errors
