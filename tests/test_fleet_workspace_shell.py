"""Workspace navigation and conversation controls against the daemon."""

from __future__ import annotations

import pytest
import test_fleet_page as fleet_page

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
no_release_lookup = fleet_page.no_release_lookup
open_page = fleet_page.open_page

pytestmark = pytest.mark.slow


def test_navigation_remembers_screen_on_reload_and_browser_back(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path="/ui/#models")
    page.wait_for_selector("#models:not([hidden])")
    assert page.locator('[href="#models"][aria-current="page"]').count() == 1
    page.click('nav a[href="#chat"]')
    page.wait_for_selector("#chat:not([hidden])")
    assert page.url.endswith("#chat")
    page.reload()
    page.wait_for_selector("#chat:not([hidden])")
    page.go_back()
    page.wait_for_selector("#models:not([hidden])")
    assert not errors


def test_signing_in_returns_to_requested_screen(joined, open_page):
    from test_fleet_ui import WORDS

    page, errors = open_page(joined, path="/ui/#models")
    page.wait_for_selector("#signin:not([hidden])")
    page.fill("#p", WORDS)
    page.click("#signin-go")
    page.wait_for_selector("#models:not([hidden])")
    assert page.url.endswith("#models")
    assert not errors


def test_conversation_search_finds_saved_message_content(joined, open_page):
    chats = joined.ui.conversations
    a = chats.start(model="sample", title="A morning conversation")
    chats.append(a.id, "user", "Tell me about lidar sensors")
    b = chats.start(model="sample", title="A different topic")
    chats.append(b.id, "user", "How do I make bread?")
    page, errors = open_page(joined, cookie=joined.cookie, path="/ui/#chat")
    page.wait_for_selector("#chat-list .chatrow", state="visible")
    page.fill("#chat-search", "lidar")
    page.wait_for_function("document.querySelectorAll('#chat-list .chatrow').length === 1")
    assert page.locator("#chat-list").inner_text().startswith("A morning conversation")
    page.click("#chat-list a")
    page.wait_for_selector(".msg.user")
    assert "lidar sensors" in page.locator("#chat-messages").inner_text()
    page.fill("#chat-search", "unmatched-term")
    page.wait_for_selector("#chat-list .hint")
    assert "No conversations match" in page.locator("#chat-list").inner_text()
    assert not errors


def test_phone_navigation_and_chat_do_not_overflow(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path="/ui/#chat")
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_selector("#chat:not([hidden])")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.locator('nav a[href="#settings"]').scroll_into_view_if_needed()
    page.click('nav a[href="#settings"]')
    page.wait_for_selector("#settings:not([hidden])")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert not errors


@pytest.fixture
def serving_chat(joined, tmp_path):
    from ml_stack.fleet.serving import Serving
    from ml_stack.testing.fakes import FakeLlamaServer, Served

    fake = FakeLlamaServer(Served(pieces=("**Hello**", " world"), gap=0.05))
    joined.ui.serving = Serving(tmp_path / "serving.json")
    joined.ui.serving.register(fake.port, [fake.served.model])
    try:
        yield fake
    finally:
        fake.close()


def test_chat_sends_history_and_renders_sanitized_markdown(joined, serving_chat, open_page):
    import json

    page, errors = open_page(joined, cookie=joined.cookie, path="/ui/#chat")
    page.wait_for_selector("#chat-askrow:not([hidden])")
    page.fill("#ask", "Hello")
    page.click("#chat-send")
    page.wait_for_selector("#chat-send:not([hidden])")
    assert page.locator(".msg.assistant strong").inner_text() == "Hello"
    page.fill("#ask", "Continue")
    page.click("#chat-send")
    page.wait_for_function("document.querySelectorAll('.msg.assistant').length === 2 && !document.querySelector('#chat-send').hidden")
    sent = [json.loads(raw) for method, path, raw in serving_chat.requests
            if method == "POST" and path == "/v1/chat/completions"]
    assert [m["content"] for m in sent[-1]["messages"]] == ["Hello", "**Hello** world", "Continue"]
    page.evaluate("""() => {
        const chat = document.querySelector('chat-view');
        chat.say('assistant', '<img src=x onerror="window.injected=true"><script>window.injected=true</script>');
    }""")
    assert page.locator(".msg.assistant script").count() == 0
    assert page.locator(".msg.assistant img[onerror]").count() == 0
    assert not page.evaluate("window.injected === true")
    assert not errors


def test_stop_closes_model_stream_and_releases_chat_controls(joined, serving_chat, open_page):
    from ml_stack.testing.fakes import Served

    serving_chat.served = Served(pieces=tuple("Still writing " for _ in range(80)), gap=0.1)
    page, errors = open_page(joined, cookie=joined.cookie, path="/ui/#chat")
    page.wait_for_selector("#chat-askrow:not([hidden])")
    page.fill("#ask", "Tell a long story")
    page.click("#chat-send")
    page.wait_for_function("document.querySelector('.msg.assistant .message-body')?.textContent.includes('Still writing')")
    page.click("#chat-stop")
    page.wait_for_selector("#chat-send:not([hidden])")
    assert "Generation stopped" in page.locator("#chat-note").inner_text()
    assert not page.locator("#chat-new").is_disabled()
    assert serving_chat.disconnected.wait(4), "The upstream model stream stayed connected"
    assert not errors
