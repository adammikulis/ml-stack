"""Shared team and model conversations driven through the Fleet page."""

import pytest
from test_fleet_chat_browser import chat_browser  # noqa: F401
from workspace_kit import Kit, clean_env

from ml_stack.workspace import tokens

pytestmark = pytest.mark.slow


def test_shared_sidebar_routes_channels_dms_threads_and_saved_model_chats(chat_browser, monkeypatch, tmp_path):
    from playwright.sync_api import expect

    kit = Kit(clean_env(monkeypatch, tmp_path / 'workspace'))
    kit.limits(sends_per_window=1000)
    worker = kit.agent('research-helper')
    tokens.store(kit.base, tokens.OWNER_FILE, kit.owner)
    kit.ws.board.create(worker, '#experiments')
    kit.ws.send(worker, '#experiments', 'note', 'The evaluation is ready.', subject='Model evaluation')
    kit.ws.send(worker, 'owner', 'note', 'Review the experiment criteria.')
    served, page = chat_browser
    saved = served.ui.conversations.start(model='model-a', title='Experiment plan')
    served.ui.conversations.append(saved.id, 'assistant', 'Compare the two evaluation runs.')
    page.goto(f'http://127.0.0.1:{served.port}/ui#chat')
    expect(page.locator('#conversation-channels')).to_contain_text('#experiments')
    sidebar = page.locator('chat-view .chats')
    sidebar.get_by_role('button', name='#experiments').click()
    expect(page.locator('#conversation-team')).to_be_visible()
    expect(page.locator('#conversation-model')).to_be_hidden()
    board = page.locator('chat-view ml-board')
    expect(board.locator('.msg pre')).to_contain_text('The evaluation is ready.')
    expect(board.locator('nav')).to_be_hidden()
    editor = board.get_by_role('textbox', name='Message', exact=True)
    editor.fill('Use the held-out dataset.')
    board.get_by_role('button', name='Send', exact=True).click()
    expect(board.locator('.msg pre')).to_contain_text(['The evaluation is ready.', 'Use the held-out dataset.'])
    assert any(message['body'] == 'Use the held-out dataset.' for message in kit.ws.board.ui_read(kit.owner, '#experiments')['messages'])
    board.get_by_role('button', name='Reply in thread').first.click()
    expect(board.get_by_role('button', name='Back to #experiments')).to_be_visible()
    sidebar.get_by_role('button', name='research-helper', exact=True).click()
    expect(board.locator('.msg pre')).to_contain_text('Review the experiment criteria.')
    sidebar.get_by_role('link', name='Experiment plan', exact=True).click()
    expect(page.locator('#conversation-model')).to_be_visible()
    expect(page.locator('#chat-messages')).to_contain_text('Compare the two evaluation runs.')
    expect(sidebar).to_be_visible()
    page.screenshot(path='/private/tmp/poolside-unified-conversations.png', full_page=True)
