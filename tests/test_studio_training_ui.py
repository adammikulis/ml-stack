"""Guided Studio training controls driven in the browser."""

import pytest
import test_fleet_page as fleet_page
from playwright.sync_api import expect

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
no_release_lookup = fleet_page.no_release_lookup
open_page = fleet_page.open_page
pytestmark = pytest.mark.slow


def test_training_steps_validate_dataset_and_preserve_recipe_arguments(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#training')
    expect(page.get_by_label('Dataset path (relative to files root)')).to_be_visible()
    page.get_by_role('button', name='Continue to model & recipe').click()
    expect(page.locator('training-view .status')).to_contain_text('Choose a dataset')
    page.get_by_label('Dataset path (relative to files root)').fill('datasets/demo.jsonl')
    page.get_by_role('button', name='Continue to model & recipe').click()
    page.get_by_label('Recipe', exact=True).select_option('tool-calls')
    page.get_by_label('Base model ID or local directory').fill('models/demo-base')
    page.get_by_label('Fine-tuning strategy').select_option('adapter')
    page.get_by_role('button', name='Review this run').click()
    expect(page.get_by_role('button', name='Queue 20-step training smoke', exact=True)).to_be_visible()
    page.get_by_role('button', name='Review command', exact=True).click()
    expect(page.locator('training-view .status')).to_contain_text('trains for 20 steps')
    spec = page.evaluate("document.querySelector('training-view').spec()")
    from ml_stack.train.run import _parser
    parsed = _parser().parse_args(spec['args'])
    assert parsed.recipe == 'tool-calls' and parsed.lora
    assert parsed.data == 'datasets/demo.jsonl' and 'base=models/demo-base' in parsed.set
    page.get_by_role('button', name='Back', exact=True).click()
    assert page.get_by_label('Base model ID or local directory').input_value() == 'models/demo-base'
    page.get_by_label('Fine-tuning strategy').select_option('full')
    parsed = _parser().parse_args(page.evaluate("document.querySelector('training-view').spec()")['args'])
    assert not parsed.lora and 'lora=false' in parsed.set
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors


def test_dataset_handoff_opens_model_step_and_keeps_run_summary(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#training')
    page.evaluate("sessionStorage.setItem('ml-stack-dataset','datasets/chosen.jsonl')")
    page.goto(f'http://127.0.0.1:{joined.port}/ui/#training')
    expect(page.get_by_label('Recipe', exact=True)).to_be_visible()
    expect(page.locator('training-view #run-summary')).to_contain_text('datasets/chosen.jsonl')
    page.get_by_role('button', name='Back', exact=True).click()
    assert page.get_by_label('Dataset path (relative to files root)').input_value() == 'datasets/chosen.jsonl'
    assert not errors
