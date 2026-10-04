"""SDK chat endpoint authority, alias and cancellation contracts."""

import asyncio
import json

import httpx
import pytest

from ml_stack.client.sdk import FleetTransport, local_client
from ml_stack.fleet.chat import Target, reply_text
from ml_stack.fleet.sdk_chat import stream
from ml_stack.testing.fakes import FakeLlamaServer, Served


@pytest.mark.redteam
def test_sdk_cannot_change_selected_peer_endpoint(monkeypatch):
    touched = []
    monkeypatch.setattr("ml_stack.client.sdk.open_stream", lambda *a, **k: touched.append(a))
    transport = FleetTransport("http://127.0.0.1:4567/infer/v1/chat/completions", "secret")
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", json={})
    with pytest.raises(ValueError, match="selected model endpoint"):
        asyncio.run(transport.handle_async_request(request))
    assert touched == []


def test_sdk_explicit_peer_prefix_and_no_environment_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unselected.invalid")
    client = local_client(Target("display", "http://127.0.0.1:4567/infer/v1/chat/completions"),
                          "server-alias")
    assert str(client.base_url) == "http://127.0.0.1:4567/infer/v1/"
    assert client.api_key != "environment-key"
    assert client.max_retries == 0
    asyncio.run(client.close())


def test_sdk_stream_alias_and_consumer_cancellation_closes_response(monkeypatch):
    from ml_stack.client import sdk

    opened = []
    original = sdk.open_stream

    def recording(*args, **kwargs):
        response = original(*args, **kwargs)
        opened.append(response)
        return response

    monkeypatch.setattr(sdk, "open_stream", recording)
    fake = FakeLlamaServer(Served(model="server-alias", pieces=("first", "second"), gap=0.05))
    try:
        target = Target("display", f"http://127.0.0.1:{fake.port}/v1/chat/completions",
                        alias="server-alias")
        pieces = stream(target, {"model": target.alias, "messages": [{"role": "user", "content": "hi"}]})
        first = next(pieces)
        assert reply_text(first) == "first"
        pieces.close()
        assert opened and opened[0].closed
        assert json.loads(fake.requests[-1][2])["model"] == "server-alias"
    finally:
        fake.close()


@pytest.mark.slow
def test_browser_sends_sdk_chat_and_reloads_saved_answer(tmp_path, playwright):
    from playwright.sync_api import expect
    from test_fleet_ui import Serving

    from ml_stack.fleet.conversations import Conversations
    from ml_stack.fleet.serving import Serving as ModelsServing

    fake = FakeLlamaServer(Served(model="qwen-sdk.gguf", pieces=("A live ", "SDK answer"), gap=0.1))
    served = Serving(tmp_path)
    served.ui.settings.setup_done = True
    served.ui.serving = ModelsServing(tmp_path / "models.json")
    served.ui.serving.register(fake.port, ["qwen-sdk.gguf"])
    served.ui.conversations = Conversations(tmp_path / "chats")
    browser = playwright.chromium.launch(headless=True)
    try:
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{served.port}/ui#chat")
        page.get_by_role("textbox", name="Message", exact=True).fill("Say hello")
        page.get_by_role("textbox", name="Message", exact=True).press("Enter")
        expect(page.locator("chat-view #chat-messages")).to_contain_text("A live SDK answer")
        page.reload()
        expect(page.locator("chat-view #chat-messages")).to_contain_text("A live SDK answer")
        assert fake.sent_to("/v1/chat/completions")[0]["model"] == "qwen-sdk.gguf"
    finally:
        browser.close()
        served.close()
        fake.close()


def test_disconnect_cancels_while_upstream_is_waiting_for_next_token():
    import socket
    import time

    fake = FakeLlamaServer(Served(model="qwen", pieces=("first", "late"), gap=10))
    browser, handler = socket.socketpair()
    pieces = stream(Target("qwen", f"http://127.0.0.1:{fake.port}/v1/chat/completions"),
                    {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
                    connection=handler)
    try:
        assert reply_text(next(pieces)) == "first"
        browser.close()
        began = time.monotonic()
        assert list(pieces) == []
        assert time.monotonic() - began < 1
    finally:
        pieces.close()
        browser.close()
        handler.close()
        fake.close()
