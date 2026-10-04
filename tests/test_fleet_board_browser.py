"""The integrated Board participates as its person owner in an isolated workspace."""

import pytest
from test_fleet_board_routes import board as board

pytestmark = pytest.mark.slow


def test_person_channel_thread_dm_drafts_and_live_reply(board, playwright):  # noqa: F811
    from playwright.sync_api import expect

    server, kit = board
    server.ui.settings.setup_done = True
    kit.ws.send(kit.worker, '#general', 'note', '<img src=x> Native sensors are ready.')
    with playwright.chromium.launch(headless=True) as browser:
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{server.port}/ui/#board')
        viewer = page.locator('board-view ml-board')
        expect(viewer.get_by_text('<img src=x> Native sensors are ready.', exact=True)).to_be_visible()
        assert viewer.locator('img').count() == 0
        viewer.get_by_role('button', name='Reply in thread', exact=True).click()
        viewer.locator('textarea').fill('Please check the stop sign.')
        viewer.get_by_role('button', name='Send', exact=True).click()
        expect(viewer.get_by_text('Please check the stop sign.', exact=True)).to_be_visible()
        rows = kit.ws.board.ui_read(kit.owner, '#general')['messages']
        assert rows[-1]['from'] == 'owner' and rows[-1]['thread'] == rows[0]['thread']
        viewer.get_by_role('button', name='Back to #general', exact=True).click()
        viewer.locator('textarea').fill('Unsent channel draft')
        viewer.get_by_role('button', name='builder', exact=True).click()
        expect(viewer.locator('textarea')).to_have_value('')
        viewer.locator('textarea').fill('Inspect the camera frames.')
        viewer.locator('textarea').press('Control+Enter')
        expect(viewer.get_by_text('Inspect the camera frames.', exact=True)).to_be_visible()
        assert kit.ws.inbox(kit.worker)[-1]['from'] == 'owner'
        kit.ws.send(kit.worker, 'owner', 'note', 'Camera frames checked.')
        expect(viewer.get_by_text('Camera frames checked.', exact=True)).to_be_visible(timeout=10000)
        viewer.get_by_role('button', name='Mark read', exact=True).click()
        page.wait_for_function("document.querySelector('ml-board').dms.every(dm => !dm.unread)")
        viewer.get_by_role('button', name='#general', exact=False).click()
        expect(viewer.locator('textarea')).to_have_value('Unsent channel draft')
        page.reload()
        expect(viewer.locator('textarea')).to_have_value('Unsent channel draft')
        assert kit.owner not in page.content() and kit.worker not in page.content()
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path='/private/tmp/ml-stack-board-audit-mobile.png', full_page=True)
        page.evaluate("window.fleetModel.go('training')")
        expect(viewer).to_have_count(0)
        assert not errors
