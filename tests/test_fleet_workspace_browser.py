"""Dataset and specialist workflow controls driven against the daemon."""

from __future__ import annotations

import pytest
from test_fleet_ui import Serving

pytestmark = pytest.mark.slow


def test_dataset_upload_preview_and_specialist_help(tmp_path, monkeypatch):
    pw = pytest.importorskip('playwright.sync_api')
    from ml_stack.fleet import page as fleet_page, routes
    from ml_stack.ui import assemble

    shell = (fleet_page.WEB / 'shell.html').read_text()
    shell = shell.replace('<fit-view></fit-view>', '<fit-view></fit-view><data-view></data-view>'
                          '<training-view></training-view><tools-view></tools-view>'
                          '<benchmarks-view></benchmarks-view><gym-view></gym-view>'
                          '<button id="test-data" onclick="fleetModel.go(\'data\')">Data workflow</button>'
                          '<button id="test-tools" onclick="fleetModel.go(\'tools\')">Tools workflow</button>')
    def render(parts):
        markup = assemble(shell, fleet_page.components(parts))
        return markup.replace('const IN_APP = [', 'const IN_APP = ["data","training","tools","gym","benchmarks",')

    monkeypatch.setattr(routes, 'render', render)
    served = Serving(tmp_path)
    served.ui.settings.setup_done = True
    try:
        with pw.sync_playwright() as api:
            browser = api.chromium.launch(headless=True)
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{served.port}/ui')
            page.locator('#test-data').click()
            page.get_by_label('Destination relative to files root').fill('datasets/demo.jsonl')
            page.get_by_label('Or paste dataset content').fill('{"label":"stop"}\n')
            page.get_by_role('button', name='Upload', exact=True).click()
            page.get_by_role('button', name='demo.jsonl', exact=True).click()
            pw.expect(page.locator('data-view #preview')).to_contain_text('"label":"stop"')
            page.get_by_role('button', name='Use for training', exact=True).click()
            assert page.get_by_label('Dataset path (relative to files root)').input_value() == 'datasets/demo.jsonl'
            page.locator('#test-tools').click()
            page.get_by_label('Installed command').select_option('ml-stack-doctor')
            page.get_by_role('button', name='Review', exact=True).click()
            pw.expect(page.locator('tools-view #runner pre')).to_contain_text('ml-stack-doctor --help')
            args = page.evaluate("document.querySelector('training-view').spec()")
            from ml_stack.train.run import _parser
            parsed = _parser().parse_args(args['args'])
            assert parsed.recipe == 'text-lm'
            assert parsed.data == 'datasets/demo.jsonl'
            assert not errors
            browser.close()
    finally:
        served.close()
