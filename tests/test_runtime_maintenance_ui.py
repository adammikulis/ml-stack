"""Healthy chat stays clear while advanced Maintenance reaches runtime repair."""

import pytest
from browser_expect import expect
from rail import reach
from test_fleet_ui import Serving

from ml_stack import agent_dependency
from ml_stack.fleet import runtime_repair
from ml_stack.fleet.conversations import Conversations
from ml_stack.fleet.serving import Serving as ModelsServing
from ml_stack.fleet.setup_jobs import Jobs
from ml_stack.scrape.browser import Window, browser
from ml_stack.testing.fakes import FakeLlamaServer, Served


@pytest.mark.slow
def test_healthy_chat_and_advanced_maintenance_repair_use_saved_preferences(tmp_path, playwright, monkeypatch):
    monkeypatch.setattr(agent_dependency, "problem", lambda: "")
    monkeypatch.setattr(runtime_repair, "resume", lambda *args: None)
    model = FakeLlamaServer(Served(model="qwen3.8-maintenance.gguf", pieces=("hello",)))
    served = Serving(tmp_path)
    served.ui.root = tmp_path
    served.ui.setup_jobs = Jobs(tmp_path)
    served.ui.settings.setup_done = True
    served.ui.serving = ModelsServing(tmp_path / "models.json")
    served.ui.serving.register(model.port, ["qwen3.8-maintenance.gguf"])
    served.ui.conversations = Conversations(tmp_path / "chats")
    try:
        with browser(Window(profile=tmp_path / "maintenance-browser", channel="chromium"), playwright) as page:
            page.goto(f"http://127.0.0.1:{served.port}/ui/#chat", wait_until="domcontentloaded")
            expect(page.locator("chat-view #chat-send")).to_be_enabled()
            page.wait_for_function("() => document.querySelector('chat-view').runtimeChecking === false")
            expect(page.locator("chat-view #chat-note")).to_be_empty()
            expect(page.get_by_role("button", name="Repair agent runtime", exact=True)).to_have_count(0)
            reach(page, 'settings')
            settings = page.locator("settings-view")
            settings.get_by_role("tab", name="Maintenance", exact=True).click()
            advanced = settings.locator("[data-pane=maintenance] details[data-advanced]")
            expect(advanced).not_to_have_attribute("open", "")
            repair = settings.get_by_role("button", name="Repair agent runtime", exact=True)
            expect(repair).to_be_hidden()
            advanced.locator("summary").click()
            expect(repair).to_be_enabled()
            settings.get_by_role("tab", name="Appearance & advanced", exact=True).click()
            settings.get_by_label("Always show advanced options", exact=True).check()
            settings.get_by_role("button", name="Save preferences", exact=True).click()
            expect(settings.locator("#settings-note")).to_contain_text("Preferences saved")
            assert served.ui.settings.always_show_advanced is True
            settings.get_by_role("tab", name="Maintenance", exact=True).click()
            expect(advanced).to_have_attribute("open", "")
            repair.click()
            expect(repair).to_be_disabled()
            expect(settings.locator("#settings-agent-runtime")).to_contain_text("Waiting to install")
            row = served.ui.setup_jobs.all()[0]
            assert row["request"] == {"repair": True} and row["state"] == "waiting"
            assert row["provenance"]["authentication"] == "ticket"
            reach(page, 'chat')
            expect(page.locator("chat-view #chat-note")).to_contain_text("Waiting to install")
            expect(page.locator("chat-view").get_by_role("button", name="Repair agent runtime", exact=True)).to_be_disabled()
            assert not model.sent_to("/v1/chat/completions")
    finally:
        served.close()
        model.close()
