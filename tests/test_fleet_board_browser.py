"""The integrated Board participates as its person owner in an isolated workspace."""

import pytest
from test_fleet_board_routes import board as board

pytestmark = pytest.mark.slow


def open_device_board(page, server):
    """The conversations screen on the device-local workspace, and the board it hosts."""
    page.goto(f'http://127.0.0.1:{server.port}/ui/#board')
    page.get_by_label('Workspace', exact=True).select_option('local')
    return page.locator('chat-view ml-board')


def open_direct(page, agent):
    """Start (or reopen) a direct conversation from the sidebar's agent directory."""
    direct = page.locator('#conversation-direct')
    if not direct.get_by_label('Message an agent', exact=True).is_visible():
        direct.locator('summary').click()
    direct.get_by_label('Message an agent', exact=True).select_option(agent)
    direct.get_by_role('button', name='Open', exact=True).click()


def test_person_channel_thread_dm_drafts_and_live_reply(board, playwright):  # noqa: F811
    from playwright.sync_api import expect

    server, kit = board
    server.ui.settings.setup_done = True
    kit.ws.send(kit.worker, '#general', 'note', '<img src=x> Native sensors are ready.')
    with playwright.chromium.launch(headless=True) as browser:
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        viewer = open_device_board(page, server)
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
        open_direct(page, kit.ws.auth(kit.worker).id)
        expect(viewer.locator('textarea')).to_have_value('')
        viewer.locator('textarea').fill('Inspect the camera frames.')
        viewer.locator('textarea').press('Control+Enter')
        expect(viewer.get_by_text('Inspect the camera frames.', exact=True)).to_be_visible()
        assert kit.ws.inbox(kit.worker)[-1]['from'] == 'owner'
        kit.ws.send(kit.worker, 'owner', 'note', 'Camera frames checked.')
        expect(viewer.get_by_text('Camera frames checked.', exact=True)).to_be_visible(timeout=10000)
        viewer.get_by_role('button', name='Mark read', exact=True).click()
        page.wait_for_function("() => document.querySelector('ml-board').dms.every(dm => !dm.unread)")
        page.locator('#conversation-channels').get_by_role('button', name='#general', exact=False).click()
        expect(viewer.locator('textarea')).to_have_value('Unsent channel draft')
        page.reload()
        expect(page.get_by_label('Workspace', exact=True)).to_have_value('local')
        expect(viewer.locator('textarea')).to_have_value('Unsent channel draft')
        assert kit.owner not in page.content() and kit.worker not in page.content()
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path='/private/tmp/poolhouse-board-audit-mobile.png', full_page=True)
        page.evaluate("window.fleetModel.go('training')")
        expect(viewer).to_be_hidden()
        assert not errors


def test_board_directory_shows_device_os_and_provenance_without_granting_identity(board, playwright):  # noqa: F811
    from playwright.sync_api import expect

    server, kit = board
    server.ui.settings.setup_done = True
    kit.ws.registry._record_device('builder', {'device_id': 'a' * 64, 'hostname': 'Mac studio',
                                 'os': 'macOS', 'source': 'local-runtime', 'verification': 'local-observed'})
    with playwright.chromium.launch(headless=True) as browser:
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        viewer = open_device_board(page, server)
        directory = page.locator('#conversation-direct')
        directory.locator('summary').click()
        entry = directory.get_by_label('Message an agent', exact=True).locator('option', has_text='Mac')
        expect(entry).to_have_count(1)
        agent = kit.ws.auth(kit.worker).id
        assert entry.get_attribute('value') == agent
        open_direct(page, agent)
        expect(viewer.locator('textarea')).to_be_visible()
        assert page.evaluate("document.querySelector('chat-view ml-board').view.b") == agent
        assert kit.worker not in page.content()
