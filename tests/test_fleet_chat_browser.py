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
    expect(page.locator("chat-view #chat-title")).to_have_text("New conversation")
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
    assert page.locator("chat-view textarea:visible").count() == 1
    expect(page.locator("chat-view ml-composer .composer-recipient")).to_have_text("Message model-a")
    assert not page.locator("chat-view #chat-options").evaluate("node => node.open")
    page.reload()
    expect(page.locator("chat-view #chat-messages")).to_contain_text("Saved answer")


def test_shared_composer_cancel_and_failed_send_keep_the_next_draft(chat_browser):
    from playwright.sync_api import expect
    served, page = chat_browser
    _open(served, page)
    page.evaluate("""() => {
      const fetch = window.fetch;
      window.fetch = (url, options) => {
        if (url !== '/ui/chat' || options?.method !== 'POST') return fetch(url, options);
        return new Promise((resolve, reject) => {
          window.failComposerSend = () => resolve(new Response(JSON.stringify({error:'Try again'}), {status:503}));
          options.signal.addEventListener('abort', () => reject(new DOMException('Stopped','AbortError')));
        });
      };
    }""")
    composer = page.get_by_role("textbox", name="Message", exact=True)
    composer.fill("First request")
    page.locator("chat-view #chat-send").click()
    expect(page.locator("chat-view #chat-cancel")).to_be_visible()
    composer.fill("Next draft")
    page.locator("chat-view #chat-cancel").click()
    expect(page.locator("chat-view #chat-note")).to_have_text("Generation stopped.")
    expect(composer).to_have_value("Next draft")
    page.evaluate("window.failComposerSend = null")
    composer.press("Enter")
    page.wait_for_function("() => typeof window.failComposerSend === 'function' && document.querySelector('chat-view').request !== null")
    composer.fill("Keep this draft")
    page.evaluate("window.failComposerSend()")
    expect(page.locator("chat-view #chat-note")).to_contain_text("Try again")
    expect(composer).to_have_value("Keep this draft")


def test_saved_temperature_restores_when_switching_and_reloading(chat_browser):
    from playwright.sync_api import expect
    served, page = chat_browser
    served.ui.conversations.start(model="model-a", title="Low temperature", settings={"temperature": .2})
    served.ui.conversations.start(model="model-a", title="Default temperature")
    _open(served, page)
    page.get_by_role("link", name="Low temperature", exact=True).click()
    page.locator("chat-view #conversation-details").click()
    expect(page.get_by_label("Temperature", exact=True)).to_have_value("0.2")
    page.get_by_label("Temperature", exact=True).fill("0.6")
    page.get_by_label("Temperature", exact=True).press("Tab")
    page.get_by_role("link", name="Default temperature", exact=True).click()
    expect(page.get_by_label("Temperature", exact=True)).to_have_value("")
    page.get_by_role("link", name="Low temperature", exact=True).click()
    expect(page.get_by_label("Temperature", exact=True)).to_have_value("0.6")
    page.reload()
    page.locator("chat-view #conversation-details").click()
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
    page.locator("chat-view").get_by_label("Search models", exact=True).fill("Qwen")
    expect(page.locator("chat-view .chat-model-choice")).to_have_count(1)
    page.locator("chat-view").get_by_label("Search models", exact=True).press("Enter")
    expect(page.locator("chat-view #chat-model-dialog")).not_to_be_visible()
    expect(page.locator("chat-view #model")).to_have_value("model-a")
    page.locator("chat-view #chat-model-button").click()
    page.locator("chat-view").get_by_label("Search models", exact=True).fill("no matches")
    expect(page.locator("chat-view #chat-model-list p")).to_have_text("No models match your search.")
    expect(page.locator("chat-view #chat-model-list").get_by_role("button", name="Browse model library")).to_be_visible()
    page.locator("chat-view").get_by_label("Search models", exact=True).press("Escape")
    expect(page.locator("chat-view #chat-model-dialog")).not_to_be_visible()
    assert page.locator("chat-view textarea:visible").count() == 1


def test_installed_model_picker_remains_visible_with_no_loaded_server(chat_browser):
    from playwright.sync_api import expect

    served, page = chat_browser
    page.unroute("**/ui/chat")
    page.route("**/ui/chat", lambda route: route.fulfill(json={"models": []}))
    page.route("**/ui/models", lambda route: route.fulfill(json={"ok": True, "library": [
        {"path": "/models/qwen.gguf", "name": "Qwen3.8-27B", "family": "Qwen", "servable": True}]}))
    _open(served, page)
    page.locator("chat-view #chat-model-button").click()
    expect(page.get_by_role("button", name="Qwen3.8-27B Installed Configure in Models →", exact=True)).to_be_visible()
    expect(page.locator("chat-view #chat-askrow")).to_be_hidden()


def test_learning_prompts_preserve_draft_and_team_channels_are_connected(chat_browser):
    from playwright.sync_api import expect

    served, page = chat_browser
    _open(served, page)
    expect(page.locator("chat-view .chat-welcome strong")).to_have_text("Start a conversation")
    expect(page.locator("chat-view .conversation-space small")).to_have_count(0)
    page.screenshot(path="/private/tmp/poolside-conversations-welcome.png", full_page=True)
    page.get_by_role("button", name="Plan an experiment", exact=True).click()
    composer = page.get_by_role("textbox", name="Message", exact=True)
    expect(composer).to_have_value("Help me design a reproducible training experiment. Ask me about the model, data, and success criteria.")
    assert not served.ui.conversations.all()
    page.evaluate("window.fleetModel.go('board')")
    expect(page.locator("chat-view #conversation-team")).to_be_visible()
    expect(page.locator("chat-view .chats")).to_be_visible()
    page.get_by_role("button", name="● model-a", exact=True).click()
    expect(composer).to_have_value("Help me design a reproducible training experiment. Ask me about the model, data, and success criteria.")
    page.screenshot(path="/private/tmp/poolside-conversations-desktop.png", full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    expect(page.locator("chat-view .chats")).to_be_hidden()
    page.get_by_role("button", name="Channels and model chats ▾").click()
    expect(page.locator("chat-view .chats")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.screenshot(path="/private/tmp/poolside-conversations-mobile.png", full_page=True)


def test_saved_model_switch_updates_visible_picker(chat_browser):
    from playwright.sync_api import expect

    served, page = chat_browser
    served.ui.conversations.start(model="offline-model", title="Offline model chat")
    _open(served, page)
    page.get_by_role("link", name="Offline model chat", exact=True).click()
    expect(page.locator("chat-view #chat-model-button")).to_have_text("offline-model — unavailable")
    expect(page.locator("chat-view #chat-send")).to_be_disabled()


def test_output_token_control_uses_backend_defaults_and_accepts_no_cap(chat_browser):
    from playwright.sync_api import expect

    served, page = chat_browser
    page.route("**/ui/conversations/defaults", lambda route: route.fulfill(json={"settings": {
        "mode": "chat", "max_output_tokens": 13579, "context": 0, "draft": "auto"}}))
    _open(served, page)
    page.locator("chat-view #conversation-details").click()
    cap = page.get_by_label("Maximum output tokens", exact=True)
    expect(cap).to_have_value("13579")
    cap.fill("")
    cap.press("Tab")
    assert page.locator("chat-view").evaluate("node => node.settings().max_output_tokens") is None


def test_installed_model_selection_hands_off_the_selected_model(chat_browser):
    from playwright.sync_api import expect

    served, page = chat_browser
    page.route("**/ui/models", lambda route: route.fulfill(json={"library": [
        {"path": "/models/installed.gguf", "name": "Installed model", "family": "Qwen", "servable": True}]}))
    page.add_init_script("window.addEventListener('fleet-select-model', e => { window.modelHandoff = e.detail; });")
    _open(served, page)
    page.locator("chat-view #chat-model-button").click()
    page.get_by_role("button", name="Installed model Installed Configure in Models →", exact=True).click()
    expect(page).to_have_url(f"http://127.0.0.1:{served.port}/ui#models")
    assert page.evaluate("window.modelHandoff") == {"path": "/models/installed.gguf", "name": "Installed model", "from": "chat"}
