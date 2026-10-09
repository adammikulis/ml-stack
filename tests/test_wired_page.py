"""The wired-memory slider on the Settings screen, driven in headless Chromium.

The privileged call is a recording stand-in; the page, the route and the estimator are real.
"""

from __future__ import annotations

import platform

import pytest

pytestmark = pytest.mark.slow

from test_fleet_page import (  # noqa: E402, F401
    browser,
    daemon,
    joined,
    no_release_lookup,
    open_page,
)
from test_wired import Recorder  # noqa: E402

from ml_stack.serve import wired, wired_apply  # noqa: E402

GIB = 1024**3


@pytest.fixture
def wired_page(joined, tmp_path, monkeypatch, open_page):  # noqa: F811
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    for name in wired_apply.AGENT_MARKERS:
        monkeypatch.delenv(name, raising=False)
    ran = Recorder(start=0)
    joined.ui.room_hooks = wired.Hooks(total=128 * GIB, runner=ran, system="Darwin",
                                       read=ran.read, daemon=tmp_path / "d.plist", others=0)
    page, errors = open_page(joined, cookie=joined.cookie)
    page.click("#nav-settings")
    page.wait_for_selector("#settings-wired .wired-slider")
    return page, errors, ran


def test_the_slider_starts_at_the_default_share_and_shows_what_is_left(wired_page):
    page, errors, _ = wired_page
    assert page.locator("#settings-wired ml-slider").get_attribute("class") == "wired-slider"
    text = page.locator("#settings-wired").inner_text()
    assert "96.0 GB" in text and "Keep this after restart" in text
    assert "before anyone logs in" in text
    assert not errors


def test_dragging_to_the_top_warns_that_the_rest_of_the_machine_may_swap(wired_page):
    page, _, _ = wired_page
    page.evaluate("""() => { const s = document.querySelector('#settings-wired ml-slider');
        s.value = s.max; s.dispatchEvent(new CustomEvent('input', {detail: {value: s.max}})); }""")
    page.wait_for_selector("#settings-wired .err:has-text('may swap')")


def test_apply_with_keep_sends_one_prompt_and_shows_the_new_state(wired_page):
    page, errors, ran = wired_page
    page.evaluate("""() => { const s = document.querySelector('#settings-wired ml-slider');
        s.value = 110592; s.dispatchEvent(new CustomEvent('input', {detail: {value: 110592}})); }""")
    page.check("#settings-wired .wired-keep")
    page.click("#settings-wired .wired-apply")
    page.wait_for_selector("#settings-wired .wired-status .ok")
    assert len(ran.calls) == 1 and "110592" in ran.calls[0][-1]
    assert "bootstrap" in ran.calls[0][-1]
    assert "108.0 GB" in page.locator("#settings-wired").inner_text()
    assert not errors


def test_back_to_default_asks_once_and_resets(wired_page):
    page, _, ran = wired_page
    ran.limit = 110592
    page.click("#settings-wired .wired-back")
    page.wait_for_selector("#settings-wired .wired-status .ok")
    assert ran.limit == 0 and len(ran.calls) == 1
