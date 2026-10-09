"""The Poolhouse page in both vocabularies, every screen, in headless Chromium against a real daemon.

Each vocabulary loads each screen with no console error and no raw catalogue id on the page. The
switch flips live from Settings, Developer, without a reload, and is kept in the browser. The
page's content security policy is the daemon's, unloosened, and the icon is served from the page's
own origin.
"""

from __future__ import annotations

import re

import pytest

#: Every test here launches headless Chromium and drives a real page.
pytestmark = pytest.mark.slow

import test_fleet_page as fleet_page  # noqa: E402
from rail import reach  # noqa: E402

from poolhouse.fleet.page_security import CONTENT_SECURITY_POLICY  # noqa: E402
from poolhouse.fleet.vocabulary_strings import CATALOGUE  # noqa: E402

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
no_release_lookup = fleet_page.no_release_lookup
open_page = fleet_page.open_page
with_peers = fleet_page.with_peers

VOCABULARIES = ("professional", "friendly")
#: each screen and the element that shows it
SCREENS = {"chat": "#chat", "cluster": "#cluster", "projects": "#projects", "settings": "#settings", "fit": "#fit"}
RAW_ID = re.compile(r"\b(?:" + "|".join(sorted({k.split(".")[0] for k in CATALOGUE})) + r")\.[a-z0-9_.]*[a-z0-9]\b")


def visit(open_page, served, vocab, route):
    """The signed-in page on one screen in one vocabulary, chosen by the address."""
    page, errors = open_page(served, path=f"/ui/?vocab={vocab}#{route}", cookie=served.cookie)
    page.wait_for_function(f"() => window.fleetModel && window.fleetModel.route === '{route}'")
    return page, errors


def vocab_of(page):
    return page.evaluate("() => document.documentElement.dataset.vocab")


@pytest.mark.parametrize("vocab", VOCABULARIES)
def test_every_screen_loads_in_both_vocabularies(with_peers, open_page, vocab):
    for route, shown in SCREENS.items():
        page, errors = visit(open_page, with_peers, vocab, route)
        assert vocab_of(page) == vocab
        assert page.title() == "Poolhouse"
        page.wait_for_selector(f"{shown}:not([hidden])")
        assert page.locator(".poolhouse-mark svg").is_visible()
        assert page.locator(".poolhouse-mark .wordmark").inner_text() == "Poolhouse"
        assert RAW_ID.findall(page.inner_text("body")) == [], route
        assert errors == [], (route, errors)


def test_the_icon_is_served_from_the_pages_own_origin_and_is_the_favicon(with_peers, open_page):
    page, _ = visit(open_page, with_peers, "professional", "cluster")
    href = page.get_attribute('link[rel="icon"]', "href")
    assert href == "/ui/static/poolhouse.svg"
    got = page.request.get(page.url.split("/ui/")[0] + href)
    assert got.status == 200 and got.headers["content-type"].startswith("image/svg+xml")
    assert b"<svg" in got.body() and b"http" not in got.body().replace(b"http://www.w3.org/2000/svg", b"")


def test_friendly_wording_keeps_the_plain_word_beside_it(with_peers, open_page):
    page, _ = visit(open_page, with_peers, "friendly", "cluster")
    heading = page.locator("cluster-view h1")
    assert heading.inner_text() == "Rooms"
    assert heading.get_attribute("data-plain") == "Devices"
    assert heading.get_attribute("title") == "Devices"
    assert "Devices" in heading.evaluate("node => getComputedStyle(node, '::after').content")
    assert page.locator('#nav-tabs a[data-workspace="pool"]').get_attribute("title") == "The Pool Deck"
    assert page.locator('#nav-context a[href="#cluster"]').inner_text() == "Rooms"


def test_the_developer_section_flips_the_vocabulary_live_and_keeps_it(with_peers, open_page):
    page, errors = visit(open_page, with_peers, "professional", "settings")
    page.evaluate("() => { window.marker = 'same page'; }")
    page.click('[data-section="developer"]')
    assert page.locator("vocabulary-settings").is_visible()
    page.check('input[name="vocab"][value="friendly"]')
    assert vocab_of(page) == "friendly"
    assert page.inner_text("settings-view h1").startswith("Household")
    assert page.evaluate("() => window.marker") == "same page"
    assert page.evaluate("() => localStorage.getItem('poolhouse.ui.vocab')") == "friendly"
    page.goto(page.url.split("?")[0].split("#")[0] + "#cluster")
    page.wait_for_function("() => window.fleetModel && window.fleetModel.route === 'cluster'")
    assert vocab_of(page) == "friendly"
    page.evaluate("() => window.fleetModel.setVocab('professional')")
    assert page.inner_text("cluster-view h1") == "Devices"
    assert errors == []


def test_the_address_wins_over_what_the_browser_kept(with_peers, open_page):
    page, _ = visit(open_page, with_peers, "friendly", "cluster")
    page.goto(page.url.split("?")[0] + "?vocab=professional#cluster")
    page.wait_for_function("() => window.fleetModel && window.fleetModel.route === 'cluster'")
    assert vocab_of(page) == "professional"


def test_the_resolution_order_is_address_then_storage_then_server_then_default(with_peers, open_page):
    page, _ = visit(open_page, with_peers, "professional", "cluster")
    pick = "(args) => window.fleetModel.vocabResolve(...args)"
    allowed = ["professional", "friendly"]
    assert page.evaluate(pick, ["friendly", "professional", "professional", "professional", allowed]) == "friendly"
    assert page.evaluate(pick, [None, "friendly", "professional", "professional", allowed]) == "friendly"
    assert page.evaluate(pick, ["", None, "friendly", "professional", allowed]) == "friendly"
    assert page.evaluate(pick, [None, None, None, "professional", allowed]) == "professional"
    assert page.evaluate(pick, ["nonsense", "also bad", "worse", "professional", allowed]) == "professional"
    assert page.evaluate(pick, [" FRIENDLY ", None, None, "professional", allowed]) == "friendly"


def test_the_page_works_when_browser_storage_is_off(browser, with_peers):
    base = f"http://127.0.0.1:{with_peers.port}"
    ctx = browser.new_context(viewport={"width": 1200, "height": 800})
    name, _, value = with_peers.cookie.partition("=")
    ctx.add_cookies([{"name": name, "value": value, "url": base}])
    ctx.add_init_script("Object.defineProperty(window, 'localStorage', {get() { throw new Error('off'); }});")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{base}/ui/?vocab=friendly#cluster")
    page.wait_for_function("() => window.fleetModel && window.fleetModel.route === 'cluster'")
    assert vocab_of(page) == "friendly"
    page.evaluate("() => window.fleetModel.setVocab('professional')")
    assert vocab_of(page) == "professional"
    assert errors == []
    ctx.close()


def test_the_content_security_policy_is_unchanged(with_peers, browser):
    base = f"http://127.0.0.1:{with_peers.port}"
    ctx = browser.new_context()
    response = ctx.new_page().goto(f"{base}/ui/?vocab=friendly")
    assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert "default-src 'none'" in CONTENT_SECURITY_POLICY and "connect-src 'self'" in CONTENT_SECURITY_POLICY
    ctx.close()


@pytest.mark.parametrize("vocab", VOCABULARIES)
def test_first_run_setup_loads_in_both_vocabularies_and_flips_live(daemon, open_page, vocab):
    page, errors = open_page(daemon, path=f"/ui/?vocab={vocab}")
    page.wait_for_selector("#first-run-card h1")
    assert vocab_of(page) == vocab
    if vocab == "professional":
        assert "Poolhouse setup · 1 of 9 · Device" in page.inner_text("#first-run")
        assert page.inner_text("#first-run-card h1").split("\n")[0] == "Set up this machine"
    else:
        assert "1 of 9" in page.inner_text("#first-run")
        assert page.inner_text("#first-run-card h1").split("\n")[0] == "Settle into your room"
    assert RAW_ID.findall(page.inner_text("#first-run")) == []
    page.evaluate(f"() => window.fleetModel.setVocab('{'friendly' if vocab == 'professional' else 'professional'}')")
    assert RAW_ID.findall(page.inner_text("#first-run")) == []
    assert errors == []


@pytest.mark.parametrize("vocab", VOCABULARIES)
def test_the_phone_width_has_no_sideways_scroll(with_peers, open_page, vocab):
    for route in ("cluster", "settings"):
        page, _ = visit(open_page, with_peers, vocab, route)
        page.set_viewport_size({"width": 390, "height": 800})
        page.wait_for_timeout(150)
        assert page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth + 1"), (route, vocab)


def test_the_navigation_still_reaches_settings(with_peers, open_page):
    page, errors = visit(open_page, with_peers, "professional", "cluster")
    reach(page, "settings")
    assert page.locator("#settings").is_visible()
    assert errors == []
