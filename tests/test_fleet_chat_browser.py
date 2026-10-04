"""Conversation history and composer controls driven against the daemon."""
from __future__ import annotations

import json

import pytest
from test_fleet_ui import Serving

from ml_stack.fleet.conversations import Conversations

pytestmark = pytest.mark.slow


@pytest.fixture
def chat_browser(tmp_path, playwright):
    served = Serving(tmp_path)
    served.ui.settings.setup_done = True
    served.ui.conversations = Conversations(tmp_path / "chats")
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.route("**/ui/chat", lambda route: route.fulfill(json={"models": [{"model": "model-a", "local": True}]}))
    try:
        yield served, page
    finally:
        browser.close()
        served.close()


def _open(served, page):
    page.goto(f"http://127.0.0.1:{served.port}/ui#chat")
    page.locator("chat-view #chat-pickrow").wait_for(state="visible")


def test_rename_search_reload_and_delete_saved_conversation(chat_browser):
    from playwright.sync_api import expect
    served, page = chat_browser
    conversation = served.ui.conversations.start(model="model-a", title="Car policy review")
    served.ui.conversations.append(conversation.id, "user", "Explain emergency braking")
    served.ui.conversations.append(conversation.id, "assistant", "Brake before the stop line.")
    other = served.ui.conversations.start(model="model-a", title="Warehouse route")
    _open(served, page)
    page.get_by_role("link", name="Car policy review", exact=True).click()
    expect(page.locator("chat-view #chat-title")).to_have_text("Car policy review")
    page.get_by_label("Options for Car policy review").click()
    page.get_by_role("button", name="Rename", exact=True).click()
    page.get_by_label("Conversation name").fill("Braking evaluation")
    page.get_by_role("button", name="Save name", exact=True).click()
    expect(page.locator("chat-view #chat-title")).to_have_text("Braking evaluation")
    assert served.ui.conversations.get(conversation.id).title == "Braking evaluation"
    page.reload()
    expect(page.locator("chat-view #chat-messages")).to_contain_text("Brake before the stop line.")
    expect(page.locator("chat-view #chat-title")).to_have_text("Braking evaluation")
    page.get_by_label("Search conversations").fill("emergency")
    expect(page.locator("chat-view #chat-list .chatrow")).to_have_count(1)
    page.get_by_label("Options for Braking evaluation").click()
    page.get_by_role("button", name="Delete", exact=True).click()
    page.get_by_role("button", name="Cancel", exact=True).click()
    assert served.ui.conversations.get(conversation.id) is not None
    page.get_by_label("Options for Braking evaluation").click()
    page.get_by_role("button", name="Delete", exact=True).click()
    page.get_by_role("button", name="Delete conversation", exact=True).click()
    expect(page.locator("chat-view #chat-title")).to_have_text("New chat")
    assert served.ui.conversations.get(conversation.id) is None
    assert served.ui.conversations.get(other.id) is not None


def test_one_composer_enter_sends_and_shift_enter_keeps_the_draft(chat_browser):
    from playwright.sync_api import expect
    served, page = chat_browser
    calls = []

    def model(route):
        if route.request.method == "GET":
            route.fulfill(json={"models": [{"model": "model-a", "local": True}]})
            return
        request = json.loads(route.request.post_data)
        calls.append(request)
        cid = request["conversation"]
        served.ui.conversations.append(cid, "user", request["messages"][-1]["content"])
        served.ui.conversations.append(cid, "assistant", "Saved answer")
        route.fulfill(content_type="text/event-stream", body='data: {"choices":[{"delta":{"content":"Saved answer"}}]}\n\ndata: [DONE]\n\n')

    page.unroute("**/ui/chat")
    page.route("**/ui/chat", model)
    _open(served, page)
    composer = page.get_by_role("textbox", name="Message", exact=True)
    composer.fill("First line")
    composer.press("Shift+Enter")
    composer.type("Second line")
    assert not calls
    composer.press("Enter")
    expect(page.locator("chat-view #chat-messages")).to_contain_text("Saved answer")
    assert calls[0]["messages"][-1]["content"] == "First line\nSecond line"
    assert page.locator("chat-view textarea").count() == 1
    assert not page.locator("chat-view #chat-options").evaluate("node => node.open")
    page.reload()
    expect(page.locator("chat-view #chat-messages")).to_contain_text("Saved answer")


def test_saved_temperature_restores_when_switching_and_reloading(chat_browser):
    from playwright.sync_api import expect
    served, page = chat_browser
    served.ui.conversations.start(model="model-a", title="Low temperature", settings={"temperature": .2})
    served.ui.conversations.start(model="model-a", title="Default temperature")
    _open(served, page)
    page.get_by_role("link", name="Low temperature", exact=True).click()
    page.locator("chat-view #chat-options summary").click()
    expect(page.get_by_label("Temperature", exact=True)).to_have_value("0.2")
    page.get_by_label("Temperature", exact=True).fill("0.6")
    page.get_by_label("Temperature", exact=True).press("Tab")
    page.get_by_role("link", name="Default temperature", exact=True).click()
    expect(page.get_by_label("Temperature", exact=True)).to_have_value("")
    page.get_by_role("link", name="Low temperature", exact=True).click()
    expect(page.get_by_label("Temperature", exact=True)).to_have_value("0.6")
    page.reload()
    page.locator("chat-view #chat-options summary").click()
    expect(page.get_by_label("Temperature", exact=True)).to_have_value("0.6")


def test_model_picker_groups_and_searches_loaded_and_installed_models(chat_browser):
    from playwright.sync_api import expect

    served, page = chat_browser
    page.route("**/ui/models", lambda route: route.fulfill(json={"ok": True, "library": [
        {"path": "/models/model-a.gguf", "name": "model-a", "family": "Qwen", "quantization": "Q4_K_XL", "servable": True},
        {"path": "/models/other.gguf", "name": "Other model", "family": "Gemma", "quantization": "Q8_0", "servable": True}]}))
    _open(served, page)
    page.locator("chat-view #chat-model-button").click()
    expect(page.get_by_role("heading", name="Qwen", exact=True)).to_be_visible()
    expect(page.get_by_role("heading", name="Gemma", exact=True)).to_be_visible()
    expect(page.locator("chat-view #chat-model-list")).to_contain_text("Loaded · Q4_K_XL")
    expect(page.locator("chat-view #chat-model-list")).to_contain_text("Installed · Q8_0")
    page.get_by_label("Search models", exact=True).fill("Qwen")
    expect(page.locator("chat-view .chat-model-choice")).to_have_count(1)
    page.get_by_label("Search models", exact=True).press("Enter")
    expect(page.locator("chat-view #chat-model-dialog")).not_to_be_visible()
    expect(page.locator("chat-view #model")).to_have_value("model-a")
    page.locator("chat-view #chat-model-button").click()
    page.get_by_label("Search models", exact=True).fill("no matches")
    expect(page.locator("chat-view #chat-model-list")).to_have_text("No models match your search.")
    page.get_by_label("Search models", exact=True).press("Escape")
    expect(page.locator("chat-view #chat-model-dialog")).not_to_be_visible()
    assert page.locator("chat-view textarea").count() == 1
