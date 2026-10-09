"""Optional person review quality links actual checks and immutable artifacts to bounded awards."""

from types import SimpleNamespace

import pytest
from browser_expect import expect
from taskboard_kit import board, proposed
from test_fleet_ui import Serving

from poolhouse.fleet import extension_routes
from poolhouse.memory import vault
from poolhouse.reputation.sealed import SealedGraph
from poolhouse.reputation.work import WorkLedger
from poolhouse.workspace import task_credit, task_outcomes, task_routes, tokens, work_reputation

__all__ = ['board']
pytestmark = [pytest.mark.slow, pytest.mark.redteam]


def test_advanced_person_quality_review_records_fixed_bonus_and_separate_rating(board, monkeypatch, tmp_path, playwright):
    proposed(board)
    tokens.store(board.base, tokens.OWNER_FILE, board.owner)
    monkeypatch.setattr(extension_routes, 'entry_points', lambda **kw: [
        SimpleNamespace(name='tasks', load=lambda: task_routes.route)])
    ledger = WorkLedger(SealedGraph(tmp_path / 'quality.enc', keys=vault.PassphraseKeys(lambda _: 'isolated quality key')))
    verify = task_credit.verify_task
    monkeypatch.setattr(task_outcomes.task_credit, 'verify_task',
                        lambda ws, token, ident: verify(ws, token, ident, ledger=ledger))
    server = Serving(tmp_path)
    server.ui.settings.setup_done = True
    try:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page(viewport={'width':390,'height':844})
            page.goto(f'http://127.0.0.1:{server.port}/ui/#tasks')
            viewer = page.locator('tasks-view')
            viewer.get_by_role('button',name='Inspect native simulation').click()
            viewer.get_by_text('Independently review this proposal',exact=True).click()
            advanced = viewer.get_by_text('Review quality — advanced',exact=True).locator('..')
            assert advanced.get_attribute('open') is None
            viewer.get_by_label('Independent review reason').fill('Independent replay meets acceptance')
            viewer.get_by_text('Review quality — advanced',exact=True).click()
            viewer.get_by_label('Validated result',exact=True).check()
            viewer.get_by_label('Validated result reason').fill('Replay independently reproduced')
            viewer.get_by_label('Validated result checks').select_option('Replay passes')
            viewer.get_by_label('Validated result artifacts').select_option('replay.json')
            viewer.get_by_label('Add independent ratings',exact=True).check()
            viewer.get_by_label('Quality rating',exact=True).fill('85')
            viewer.get_by_label('Reliability assessment',exact=True).fill('80')
            viewer.get_by_label('Rating reason',exact=True).fill('Reproduced evidence and clear implementation')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path='/private/tmp/poolhouse-task-quality-form.png',full_page=True)
            viewer.get_by_role('button',name='Record independent review',exact=True).click()
            expect(viewer.locator('.status')).to_contain_text('checks you independently verified')
            assert board.board.get(board.owner,board.task['id'])['review'] is None
            viewer.get_by_label('Replay passes',exact=True).check()
            viewer.get_by_role('button',name='Record independent review',exact=True).click()
            expect(viewer.locator('.badge')).to_have_text('completed')
            expect(viewer.locator('.status')).to_contain_text('15 credits')
            page.screenshot(path='/private/tmp/poolhouse-task-quality-preview.png',full_page=True)
        task = board.board.get(board.owner,board.task['id'])
        assert task['review']['quality'] == [{'kind':'validated','reason':'Replay independently reproduced',
                                             'checks':['Replay passes'],'artifacts':['replay.json']}]
        assert task['review']['review']['quality'] == 85
        standing = work_reputation.standings(board.ws,board.owner,ledger=ledger)
        earned = next(row for row in standing['team'] if row['economy']['earned'])
        assert earned['economy']['completion_credits'] == 10
        assert earned['economy']['quality_credits'] == 5
        assert earned['economy']['balance'] == 15
        assert earned['work_reputation']['quality_samples'] == 1
    finally:
        server.close()
        ledger.sealed.close()
