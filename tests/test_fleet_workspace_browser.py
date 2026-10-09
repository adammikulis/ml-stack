"""Dataset and specialist workflow controls driven against the daemon."""

from __future__ import annotations

import pytest
from rail import reach
from test_fleet_ui import Serving

pytestmark = pytest.mark.slow


def test_dataset_upload_preview_and_specialist_help(tmp_path, monkeypatch, playwright):
    pw = pytest.importorskip('playwright.sync_api')
    served = Serving(tmp_path)
    served.ui.settings.setup_done = True
    try:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{served.port}/ui')
        reach(page, 'data')
        page.get_by_label('Destination relative to files root').fill('datasets/demo.jsonl')
        page.get_by_label('Or paste dataset content').fill('{"label":"stop"}\n')
        page.get_by_role('button', name='Upload', exact=True).click()
        page.get_by_role('button', name='demo.jsonl', exact=True).click()
        pw.expect(page.locator('data-view #preview')).to_contain_text('"label":"stop"')
        page.get_by_role('button', name='Use for training', exact=True).click()
        assert page.get_by_label('Dataset path (relative to files root)').input_value() == 'datasets/demo.jsonl'
        reach(page, 'tools')
        page.get_by_role('tab', name='Command library', exact=True).click()
        page.get_by_label('Installed command').select_option('poolhouse-doctor')
        page.get_by_role('button', name='Review', exact=True).click()
        pw.expect(page.locator('tools-view #runner pre')).to_contain_text('poolhouse-doctor --help')
        args = page.evaluate("document.querySelector('training-view').spec()")
        from poolhouse.train.run import _parser
        parsed = _parser().parse_args(args['args'])
        assert parsed.recipe == 'text-lm'
        assert parsed.data == 'datasets/demo.jsonl'
        check_training_controls(page, pw)
        reach(page, 'gym')
        page.get_by_label('Controller', exact=True).select_option('ppo')
        page.get_by_label('PPO checkpoint path').fill('/tmp/policy.zip')
        pw.expect(page.get_by_role('button', name='Apply controller', exact=True)).to_be_disabled()
        applied = page.evaluate("""async () => {
            const gym = document.querySelector('gym-view');
            const calls = [];
            gym.session = 'browser-controlled';
            gym.applyController.disabled = false;
            gym.control = async (command, payload) => calls.push({command, payload});
            gym.decisionCheckpoint.value = '/tmp/trained-pointer';
            gym.controller.value = 'decider';
            gym.controller.dispatchEvent(new Event('change'));
            const before = calls.length;
            await gym.apply();
            gym.session = null;
            return {before, calls};
        }""")
        assert applied['before'] == 1
        assert applied['calls'][0]['payload']['decision_checkpoint'] == '/tmp/trained-pointer'
        assert not errors
        browser.close()
    finally:
        served.close()


def check_training_controls(page, pw):
    from poolhouse.train.run import _parser

    reach(page, 'training')
    page.get_by_label('Recipe', exact=True).select_option('tool-calls')
    assert page.get_by_label('Base model ID or local directory').count() == 1
    page.get_by_label('Base model ID or local directory').fill('models/demo-base')
    pw.expect(page.get_by_role('button', name='Queue 20-step training smoke', exact=True)).to_be_visible()
    page.get_by_role('button', name='Review command', exact=True).click()
    pw.expect(page.locator('training-view #config .status')).to_contain_text('trains for 20 steps')
    page.get_by_label('Fine-tuning strategy').select_option('adapter')
    specification = page.evaluate("document.querySelector('training-view').spec()")
    parsed = _parser().parse_args(specification['args'])
    assert parsed.lora and 'base=models/demo-base' in parsed.set
    page.get_by_label('Fine-tuning strategy').select_option('full')
    specification = page.evaluate("document.querySelector('training-view').spec()")
    parsed = _parser().parse_args(specification['args'])
    assert not parsed.lora and 'lora=false' in parsed.set


def test_short_views_start_below_context_navigation(tmp_path, playwright):
    served = Serving(tmp_path)
    served.ui.settings.setup_done = True
    try:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.goto(f'http://127.0.0.1:{served.port}/ui')
        for route in ('fit', 'knowledge', 'tools'):
            page.evaluate("(route)=>location.hash=route", route)
            page.wait_for_function("(route)=>window.fleetModel.route===route", arg=route)
            header = page.locator('.context-navigation').bounding_box()
            workspace = page.locator('#workspace').bounding_box()
            assert header and workspace
            assert abs(workspace['y'] - header['y'] - header['height']) < 2
        browser.close()
    finally:
        served.close()
