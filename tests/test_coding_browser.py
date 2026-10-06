"""Shared composer Coding lifecycle through real routes and native subprocesses."""
from __future__ import annotations

import pytest
from coding_kit import fixture_worker
from test_fleet_ui import Serving
from workspace_kit import Kit

from ml_stack.fleet.conversations import Conversations
from ml_stack.workspace import coding_turns

pytestmark = pytest.mark.slow


@pytest.fixture
def coding_browser(tmp_path, monkeypatch, playwright):
    monkeypatch.setenv("ML_STACK_WORKSPACE_HOME", str(tmp_path / "workspace"))
    Kit(tmp_path / "workspace")
    monkeypatch.setattr(coding_turns, "worker", fixture_worker)
    served = Serving(tmp_path)
    served.ui.settings.setup_done = True
    served.ui.conversations = Conversations(tmp_path / "chats")
    project = tmp_path / "project"
    project.mkdir()
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.route("**/ui/chat", lambda route: route.fulfill(json={"models": [{"model": "chat-model", "local": True}]}))
    page.route("**/ui/coding/catalogue", lambda route: route.fulfill(json={
        "ok": True, "harnesses": [{"name": "pi", "available": True}, {"name": "codex", "available": True}],
        "roles": [{"name": "read-only", "summary": "Inspect without editing"}],
        "default_role": "read-only", "context": 262144,
        "default_model": "test-model", "models": [{"name": "Fixture coding model", "ref": "test-model"}]}))
    try:
        yield served, page, project
    finally:
        if hasattr(served.ui, "coding_turns"):
            for turn in served.ui.coding_turns.turns.values():
                if turn.state in ("starting", "running", "cancelling"):
                    turn.cancel()
        browser.close()
        served.close()


def test_shared_composer_saves_coding_settings_and_resumes_native_session(coding_browser):
    from playwright.sync_api import expect

    served, page, project = coding_browser
    page.goto(f"http://127.0.0.1:{served.port}/ui#chat")
    page.get_by_label("Mode", exact=True).select_option("coding")
    page.get_by_label("Project directory", exact=True).fill(str(project))
    page.get_by_label("Project directory", exact=True).press("Tab")
    expect(page.locator("chat-view #model")).to_have_value("test-model")
    composer = page.get_by_role("textbox", name="Message", exact=True)
    composer.fill("inspect the brakes")
    composer.press("Enter")
    expect(page.locator("chat-view #chat-messages")).to_contain_text("Inspected: inspect the brakes")
    expect(page.locator("chat-view #chat-stop")).to_be_hidden()
    assert page.locator("chat-view textarea").count() == 1
    assert not page.locator("chat-view #chat-options").evaluate("node => node.open")
    cid = page.locator("chat-view").evaluate("node => node.open")
    saved = served.ui.conversations.get(cid)
    assert saved.settings["mode"] == "coding"
    assert saved.settings["project"] == str(project)
    assert saved.settings["harness"] == "codex"
    assert saved.settings["role"] == "read-only"
    page.reload()
    expect(page.get_by_label("Mode", exact=True)).to_have_value("coding")
    expect(page.get_by_label("Project directory", exact=True)).to_have_value(str(project))
    composer.fill("inspect the sensors")
    composer.press("Enter")
    expect(page.locator("chat-view #chat-messages")).to_contain_text("Inspected: inspect the sensors")
    expect(page.locator("chat-view #chat-stop")).to_be_hidden()
    assert len(served.ui.conversations.get(cid).messages) == 4


def test_reload_reattaches_running_coding_turn_and_stop_keeps_the_conversation(coding_browser):
    from playwright.sync_api import expect

    served, page, project = coding_browser
    saved = served.ui.conversations.start(model="test-model", title="Live coding", settings={
        "mode": "coding", "project": str(project), "role": "read-only"})
    page.goto(f"http://127.0.0.1:{served.port}/ui#chat")
    page.get_by_role("link", name="Live coding", exact=True).click()
    expect(page.get_by_label("Mode", exact=True)).to_have_value("coding")
    expect(page.get_by_label("Project directory", exact=True)).to_have_value(str(project))
    expect(page.locator("chat-view #model")).to_have_value("test-model")
    expect(page.locator("chat-view #chat-send")).to_be_enabled()
    composer = page.get_by_role("textbox", name="Message", exact=True)
    composer.fill("wait until cancelled")
    composer.press("Enter")
    expect(page.locator("chat-view #chat-status")).to_contain_text("running")
    page.reload()
    expect(page.locator("chat-view #chat-stop")).to_be_visible()
    page.locator("chat-view #chat-stop").click()
    expect(page.locator("chat-view #chat-stop")).to_be_hidden()
    assert served.ui.coding_turns.status(saved.id)["state"] == "cancelled"
    composer.fill("continue inspection")
    composer.press("Enter")
    expect(page.locator("chat-view #chat-messages")).to_contain_text("Inspected: continue inspection")
