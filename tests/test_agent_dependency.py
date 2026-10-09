"""Missing runtime dependencies remain unavailable before launch."""
import sys
from types import SimpleNamespace

import pytest

from ml_stack import agent_dependency
from ml_stack.fleet.chat_routes import ChatRoutes
from ml_stack.workspace import localstart


def test_missing_sdk_reports_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "agents", None)
    assert "ml-stack[agents]" in agent_dependency.problem()


def test_local_chat_refuses_before_spawn(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "agents", None)
    monkeypatch.setattr(localstart.la, "check_project", lambda *args: "")
    monkeypatch.setattr(localstart.la, "check_orders", lambda value: value)
    spawned = []
    with pytest.raises(localstart.Unavailable, match="agent runtime"):
        localstart._start(SimpleNamespace(base=tmp_path), localstart.Ask(), spawn=lambda *a, **k: spawned.append(a))
    assert not spawned


def test_chat_readiness_reports_actual_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "agents", None)
    from ml_stack.fleet import chat_routes
    monkeypatch.setattr(chat_routes, "load_cluster_key", lambda path: None)
    monkeypatch.setattr(chat_routes, "targets", lambda *args: [])
    route = ChatRoutes()
    route.ui = SimpleNamespace(cluster_key_path=None, serving=None)
    route.method = "GET"
    replies = []
    route.send = lambda status, body: replies.append((status, body))
    assert route._chat()
    assert replies[0][0] == 200
    assert replies[0][1]["runtime_ready"] is False
    assert "ml-stack[agents]" in replies[0][1]["runtime_error"]


def test_missing_chat_sdk_does_not_change_coding_prerequisites(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "agents", None)
    monkeypatch.setattr(localstart.la, "check_project", lambda *args: "")
    monkeypatch.setattr(localstart.la, "check_orders", lambda value: value)
    monkeypatch.setattr(localstart.localmodel, "choose", lambda *a, **k: SimpleNamespace(ok=False, problem="no coding model", hint="download a model"))
    with pytest.raises(localstart.Unavailable, match="no coding model"):
        localstart._start(SimpleNamespace(base=tmp_path), localstart.Ask(profile="coding"))


@pytest.mark.slow
def test_missing_sdk_disables_send_and_enter(tmp_path, playwright, monkeypatch):
    from playwright.sync_api import expect
    from test_fleet_ui import Serving

    from ml_stack.fleet.conversations import Conversations
    from ml_stack.fleet.serving import Serving as ModelsServing
    from ml_stack.testing.fakes import FakeLlamaServer, Served

    fake = FakeLlamaServer(Served(model="qwen3.8-test.gguf", pieces=("hello",)))
    served = Serving(tmp_path)
    served.ui.settings.setup_done = True
    served.ui.serving = ModelsServing(tmp_path / "models.json")
    served.ui.serving.register(fake.port, ["qwen3.8-test.gguf"])
    served.ui.conversations = Conversations(tmp_path / "chats")
    monkeypatch.setitem(sys.modules, "agents", None)
    browser = playwright.chromium.launch(headless=True)
    try:
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{served.port}/ui#chat")
        expect(page.locator("chat-view #chat-send")).to_be_disabled()
        expect(page.locator("chat-view #chat-note")).to_contain_text("agent runtime")
        page.get_by_role("textbox", name="Message", exact=True).fill("Say hello")
        page.get_by_role("textbox", name="Message", exact=True).press("Enter")
        assert not fake.sent_to("/v1/chat/completions")
    finally:
        browser.close()
        served.close()
        fake.close()
