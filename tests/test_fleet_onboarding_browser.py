"""First-launch progress and failed-save behavior in Chromium."""

import pytest
from browser_expect import expect

try:
    import playwright.sync_api  # noqa: F401
except ModuleNotFoundError:
    pytest_plugins = []

    @pytest.fixture
    def daemon():
        raise pytest.skip.Exception("needs playwright: poolhouse[scrape]", allow_module_level=False)

    @pytest.fixture
    def joined():
        raise pytest.skip.Exception("needs playwright: poolhouse[scrape]", allow_module_level=False)

    @pytest.fixture
    def open_page(daemon):
        return None
else:
    pytest_plugins = ["test_fleet_page"]

pytestmark = pytest.mark.slow


def test_setup_names_the_current_stage(daemon, open_page):
    page, errors = open_page(daemon)
    page.wait_for_selector("#first-run:not([hidden])")
    assert "1 of 9 · Device" in page.locator("#first-run-card > p").inner_text()
    page.get_by_role("button", name="Continue", exact=True).click()
    expect(page.locator("#first-run-card > p")).to_contain_text("2 of 9 · Pool")
    assert not errors


def test_failed_startup_save_stays_on_the_step(daemon, open_page, monkeypatch):
    monkeypatch.setattr(daemon.ui, "apply_prefs", lambda req: {"error": "Startup unavailable"})
    page, errors = open_page(daemon)
    page.wait_for_selector("#first-run:not([hidden])")
    page.evaluate("document.querySelector('first-run').go(3)")
    page.wait_for_selector("#autostart-manual")
    page.get_by_role("button", name="Continue", exact=True).click()
    page.wait_for_selector("#first-run [role=alert]")
    assert page.locator("#first-run [role=alert]:visible").inner_text() == "Startup unavailable"
    assert page.get_by_role("button", name="Continue", exact=True).is_enabled()
    assert "4 of 9 · Startup" in page.locator("#first-run-card > p").inner_text()
    assert not errors


def test_empty_passphrase_explains_what_to_enter(joined, open_page):
    page, errors = open_page(joined)
    page.wait_for_selector("#signin:not([hidden])")
    page.locator("#signin-go").click()
    assert page.locator("#signin-note").inner_text() == "Enter the cluster passphrase."
    assert page.locator("#signin-go").is_enabled()
    assert not errors
