"""The fleet interface, driven in headless Chromium against a real daemon.

Each test opens the page the way a person does -- click the tab, type in the box, press
the button, read what came back -- against `test_fleet_ui.Serving`, which is the daemon
on a real socket with the real routes behind it. Nothing here fakes a response.

Everything skips when Playwright or Chromium is unavailable. No test reads a data file:
the models, the conversations and the measurements are invented in ``tmp_path``.
"""

from __future__ import annotations

import pytest

pw = pytest.importorskip("playwright.sync_api")

#: Every test here launches headless Chromium and drives a real page.
pytestmark = pytest.mark.slow

from test_fleet_ui import WORDS, Serving  # noqa: E402

GIB = 1024 ** 3
ROOM = 96 * GIB


def sample_fits():
    """Two invented measurements: a small dense model and one with a fat cache."""
    from ml_stack.serve.fit import Fit

    return [
        Fit(model="thornfield-8B-Q4_K_M.gguf", weights=5 * GIB, compute=GIB, room=ROOM,
            per_token=32768, per_seq=8 * 1024 * 1024, cache_type="f16", build="a1b2c3d"),
        Fit(model="marrowgate-A3B-UD-Q4_K_XL.gguf", weights=60 * GIB, compute=2 * GIB,
            room=ROOM, per_token=4096, per_seq=0, cache_type="q8_0", build="a1b2c3d"),
    ]


@pytest.fixture(scope="session")
def browser():
    with pw.sync_playwright() as p:
        try:
            # headless is the default, said out loud: a test must never take the screen
            b = p.chromium.launch(headless=True)
        except Exception as exc:                       # noqa: BLE001
            pytest.skip(f"chromium did not launch: {exc}")
        yield b
        b.close()


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    """A daemon with a model store, a chat store and somewhere to keep things."""
    import urllib.error

    from ml_stack.fleet import models as models_mod
    from ml_stack.fleet.conversations import Conversations
    from ml_stack.fleet.models import Models

    served = Serving(tmp_path)
    served.ui.root = tmp_path / "traind"
    served.ui.conversations = Conversations(tmp_path / "chats")
    served.ui.models = Models([served.files], served.files)

    # No hub: the popular list must still draw, and a test must not wait on the internet.
    def unreachable(*a, **k):
        raise urllib.error.URLError("no network in tests")

    monkeypatch.setattr(models_mod, "_hub", unreachable)
    try:
        yield served
    finally:
        served.httpd.shutdown()
        served.httpd.server_close()


@pytest.fixture
def joined(daemon):
    """That daemon, in a cluster, with a session cookie for the browser."""
    daemon.call("/ui/setup/join", method="POST",
                body={"passphrase": WORDS, "group": "home"})
    _, _, headers = daemon.call("/ui/session", method="POST", body={"passphrase": WORDS})
    daemon.cookie = headers["Set-Cookie"].split(";")[0]
    return daemon


@pytest.fixture
def with_peers(joined):
    """One invented machine on the LAN, so the cluster view has a card to draw.

    The route still assembles the answer; only the multicast sweep is stood in for,
    because a test has no second machine.
    """
    joined.ui.peers = lambda force=False: [{
        "name": "greenhollow", "port": 8770, "host": "10.0.0.9",
        "base_url": "http://10.0.0.9:8770", "is_self": False, "clusters": ["home"],
        "slots": 4, "free": 1, "queued": 2, "busy": False,
        "device": {"vendor": "nvidia", "cuda": True, "gpu": "Marrowgate 5000",
                   "cpus": 16, "ram_gb": 64.0, "ram_used_gb": 20.0, "cpu_pct": 41.0,
                   "vram_total_gb": 24.0, "vram_free_gb": 6.0, "gpu_util_pct": 88.0,
                   "temp_c": 79.0, "clock_mhz": 2400, "power_w": 310.5,
                   "throttled": True, "labels": ["train"],
                   "models": [{"name": "thornfield-8B-Q4_K_M.gguf"}]},
    }]
    return joined


@pytest.fixture
def open_page(browser):
    """Opens the interface in a fresh context; returns ``(page, errors)``."""
    contexts = []

    def _open(served, *, path="/ui/", cookie=""):
        base = f"http://127.0.0.1:{served.port}"
        ctx = browser.new_context(viewport={"width": 1400, "height": 950})
        contexts.append(ctx)
        if cookie:
            name, _, value = cookie.partition("=")
            ctx.add_cookies([{"name": name, "value": value, "url": base}])
        page = ctx.new_page()
        page.set_default_timeout(10_000)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text)
                if m.type == "error" and "Failed to load resource" not in m.text else None)
        page.goto(base + path)
        return page, errors

    yield _open
    for ctx in contexts:
        ctx.close()


# -- first run ---------------------------------------------------------------------------
class TestFirstRun:
    def test_a_machine_in_no_cluster_opens_on_the_wizard(self, daemon, open_page):
        page, errors = open_page(daemon)
        page.wait_for_selector("#first-run:not([hidden])")
        assert page.locator("#first-run h1").inner_text() == "Set up this machine"
        assert not errors

    def test_naming_it_moves_on_to_the_clusters_step(self, daemon, open_page):
        page, errors = open_page(daemon)
        page.wait_for_selector("#first-run:not([hidden])")
        page.fill("#n", "quillhaven")
        page.click("#first-run button:has-text('Continue')")
        page.wait_for_selector("#first-run h1:has-text('Clusters')")
        assert page.locator("#p1").is_visible()
        assert not errors

    def test_a_short_passphrase_cannot_be_joined_with(self, daemon, open_page):
        page, errors = open_page(daemon)
        page.wait_for_selector("#first-run:not([hidden])")
        page.click("#first-run button:has-text('Continue')")
        page.wait_for_selector("#p1")
        page.fill("#p1", "abc")
        assert page.locator("#first-run button:has-text('Join')").is_disabled()
        page.fill("#p1", "correct horse battery")
        assert page.locator("#first-run button:has-text('Join')").is_enabled()
        assert not errors


# -- signing in --------------------------------------------------------------------------
class TestSigningIn:
    def test_a_cluster_asks_for_its_passphrase(self, joined, open_page):
        page, errors = open_page(joined)
        page.wait_for_selector("#signin:not([hidden])")
        assert "passphrase" in page.locator("#signin .sub").inner_text()
        assert not errors

    def test_the_wrong_words_are_refused_and_the_right_ones_open_the_cluster(
            self, joined, open_page):
        page, errors = open_page(joined)
        page.wait_for_selector("#signin:not([hidden])")
        page.fill("#p", "not the passphrase")
        page.click("#signin-go")
        page.wait_for_selector("#signin-note .err")
        assert "match" in page.locator("#signin-note .err").inner_text()

        page.fill("#p", WORDS)
        page.click("#signin-go")
        page.wait_for_selector("#cluster:not([hidden])")
        assert page.locator("#cluster h1").inner_text() == "Cluster"
        assert not errors


# -- the cluster -------------------------------------------------------------------------
class TestTheClusterView:
    def test_a_machine_on_its_own_says_how_to_add_another(self, joined, open_page):
        page, errors = open_page(joined, cookie=joined.cookie)
        page.wait_for_selector("#cluster-cards .empty")
        assert "Just this machine so far" in page.locator("#cluster-cards").inner_text()
        assert "MACHINES" in page.locator("#cluster-stat").inner_text().upper()
        assert not errors

    def test_a_machines_card_says_what_it_is_and_what_it_is_doing(self, with_peers,
                                                                  open_page):
        page, errors = open_page(with_peers, cookie=with_peers.cookie)
        page.wait_for_selector("#cluster-cards .peer")
        card = page.locator("#cluster-cards .peer").first.inner_text()
        assert "greenhollow" in card
        assert "NVIDIA · CUDA" in card and "Marrowgate 5000" in card
        assert "3 running of 4" in card and "2 waiting" in card
        assert "20.0 GB of 64.0 GB in use" in card
        assert "41% busy" in card and "88% busy" in card
        assert "6.0 GB free of 24.0 GB" in card
        assert "79°C" in card and "2400 MHz" in card and "310.5 W" in card
        assert "throttled" in card
        assert "3/4" in page.locator("#cluster-stat").inner_text()
        assert not errors

    def test_the_cluster_it_is_in_is_listed_with_a_way_out(self, joined, open_page):
        page, errors = open_page(joined, cookie=joined.cookie)
        page.wait_for_selector("#cluster-joined:has-text('home')")
        assert page.locator("#cluster-joined button:has-text('Leave')").count() == 1
        assert not errors

    def test_the_sweep_command_follows_what_is_ticked(self, joined, open_page):
        page, errors = open_page(joined, cookie=joined.cookie)
        page.wait_for_selector("#cluster-sweep .searchrow")
        shown = page.locator("#cluster-sweep pre.cmd").first
        assert shown.inner_text().strip() == "ml-stack-bench sweep --fleet"
        page.fill("#sample", "40")
        page.fill("#label", "nightly")
        assert "--sample 40" in shown.inner_text()
        assert "--label nightly" in shown.inner_text()
        assert not errors


# -- chat --------------------------------------------------------------------------------
class TestTheChatView:
    def test_with_nothing_serving_it_says_where_to_start_one(self, joined, open_page):
        page, errors = open_page(joined, cookie=joined.cookie)
        page.wait_for_selector("#cluster:not([hidden])")
        page.click("nav.tabs a:has-text('Chat')")
        page.wait_for_selector("#chat-none:not([hidden])")
        assert "No model is running yet" in page.locator("#chat-none").inner_text()
        assert page.locator("#chat-askrow").is_hidden()
        assert not errors

    def test_a_kept_conversation_is_listed_and_opens(self, joined, open_page):
        joined.call("/ui/conversations", method="POST", cookie=joined.cookie,
                    body={"model": "thornfield-8B", "title": "about the roof"})
        page, errors = open_page(joined, cookie=joined.cookie)
        page.click("nav.tabs a:has-text('Chat')")
        page.wait_for_selector("#chat-list .chatrow")
        assert "about the roof" in page.locator("#chat-list").inner_text()
        page.click("#chat-list .chatrow a")
        page.wait_for_selector("#chat-list .chatrow.on")
        assert not errors

    def test_deleting_one_takes_it_off_the_list(self, joined, open_page):
        joined.call("/ui/conversations", method="POST", cookie=joined.cookie,
                    body={"model": "thornfield-8B", "title": "about the roof"})
        page, errors = open_page(joined, cookie=joined.cookie)
        page.click("nav.tabs a:has-text('Chat')")
        page.wait_for_selector("#chat-list .chatrow")
        page.click("#chat-list .chatrow button:has-text('Delete')")
        page.wait_for_selector("#chat-list .chatrow", state="detached")
        assert not errors


# -- models ------------------------------------------------------------------------------
class TestTheModelsView:
    def test_a_model_on_this_machine_is_listed_with_its_size(self, joined, open_page):
        (joined.files / "thornfield-8B-Q4_K_M.gguf").write_bytes(b"x" * (2 << 20))
        page, errors = open_page(joined, cookie=joined.cookie)
        page.click("nav.tabs a:has-text('Models')")
        page.wait_for_selector("#models-here h2")
        here = page.locator("#models-here")
        assert "thornfield-8B-Q4_K_M.gguf" in here.inner_text()
        assert "GB free on this machine" in page.locator("#models-free").inner_text()
        assert not errors

    def test_with_no_hub_the_search_box_still_takes_a_query(self, joined, open_page):
        page, errors = open_page(joined, cookie=joined.cookie)
        page.click("nav.tabs a:has-text('Models')")
        page.wait_for_selector("#models-popular .hint")
        page.fill("#hunt", "thornfield")
        page.wait_for_selector("#models-popular h2:has-text('Models matching')")
        assert "THORNFIELD" in page.locator("#models-popular h2").first.inner_text().upper()
        assert not errors


# -- settings ----------------------------------------------------------------------------
class TestTheSettingsView:
    def test_it_says_what_this_machine_is_called(self, joined, open_page):
        page, errors = open_page(joined, cookie=joined.cookie)
        page.click("nav.tabs a:has-text('Settings')")
        page.wait_for_selector("#settings-sub:not(:empty)")
        assert "studio" in page.locator("#settings-sub").inner_text()
        assert not errors

    def test_choosing_a_job_and_saving_says_it_saved(self, joined, open_page):
        page, errors = open_page(joined, cookie=joined.cookie)
        page.click("nav.tabs a:has-text('Settings')")
        page.wait_for_selector("#settings-left .group")
        page.check("#labels-train\\,prep")
        page.click("#settings-save")
        page.wait_for_selector("#settings-note .ok")
        assert page.locator("#settings-note .ok").inner_text() == "Saved."
        _, got, _ = joined.call("/ui/settings", cookie=joined.cookie)
        assert sorted(got["settings"]["labels"]) == ["prep", "train"]
        assert not errors

    def test_no_screen_shows_the_word_null(self, joined, open_page):
        """`replaceChildren` writes the word "null" for a gap the way `el` never does."""
        page, errors = open_page(joined, cookie=joined.cookie)
        page.wait_for_selector("#cluster-joined .row")
        for tab, ready in (("Chat", "#chat-none, #chat-askrow"),
                           ("Models", "#models-here h2"),
                           ("Settings", "#settings-removal label.opt"),
                           ("Fit", "table.fit tbody tr"),
                           ("Cluster", "#cluster-sweep .searchrow")):
            page.click(f"nav.tabs a:has-text('{tab}')")
            page.wait_for_selector(ready)
            shown = page.locator("#root").inner_text()
            assert "\nnull" not in shown and not shown.startswith("null"), tab
        assert not errors

    def test_the_remove_section_lists_what_would_go(self, joined, open_page):
        page, errors = open_page(joined, cookie=joined.cookie)
        page.click("nav.tabs a:has-text('Settings')")
        page.wait_for_selector("#settings-removal label.opt")
        assert "cannot be undone" not in page.locator("#settings-removal").inner_text()
        page.click("#settings-removal button.danger")
        page.wait_for_selector("#settings-removal .hint:has-text('cannot be undone')")
        assert "click again" in page.locator("#settings-removal button.danger").inner_text()
        assert not errors


# -- what fits ---------------------------------------------------------------------------
class TestTheFitView:
    @pytest.fixture(autouse=True)
    def _measured(self, fit_files):
        return fit_files(sample_fits(), room=ROOM).shipped

    def test_the_table_says_what_the_command_says(self, joined, open_page):
        """Every number on the screen is `serve.fit`'s own, so the two cannot disagree."""
        from ml_stack.serve.fit import Fit

        page, errors = open_page(joined, cookie=joined.cookie)
        page.click("nav.tabs a:has-text('Fit')")
        page.wait_for_selector("table.fit tbody tr")
        rows = page.locator("table.fit tbody tr")
        assert rows.count() == 2

        _, got, _ = joined.call("/ui/fit.json?users=8", cookie=joined.cookie)
        for i, row in enumerate(got["records"]):
            here = Fit.from_dict(row).at_room(got["at_room"])
            cells = rows.nth(i).locator("td")
            assert cells.nth(1).inner_text() == f"{here.loaded() / GIB:.2f}"
            assert cells.nth(3).inner_text().replace(",", "") == str(here.users(32768))
            assert cells.nth(4).inner_text().replace(",", "") == str(here.longest(8))
        assert not errors

    def test_moving_the_room_asks_again_and_seats_fewer(self, joined, open_page):
        page, errors = open_page(joined, cookie=joined.cookie)
        page.click("nav.tabs a:has-text('Fit')")
        page.wait_for_selector("table.fit tbody tr")
        before = page.locator("table.fit tbody tr").first.locator("td").nth(3).inner_text()
        page.select_option("#fit-controls select", "8")
        page.wait_for_function(
            "([was]) => document.querySelector('table.fit tbody td:nth-child(4)')"
            ".textContent !== was", arg=[before])
        after = page.locator("table.fit tbody tr").first.locator("td").nth(3).inner_text()

        _, got, _ = joined.call(f"/ui/fit.json?room={8 * GIB}", cookie=joined.cookie)
        seats = got["records"][0]["seats"][got["ladder"].index(32768)]
        assert after == (f"{seats:,}" if seats else "does not fit")
        assert after != before
        assert "8.0G of room" in page.locator("fit-view").inner_text()
        assert not errors

    def test_both_panels_are_drawn(self, joined, open_page):
        page, errors = open_page(joined, cookie=joined.cookie)
        page.click("nav.tabs a:has-text('Fit')")
        page.wait_for_selector(".panels svg path.ln")
        assert page.locator(".panels .panel").count() == 2
        assert page.locator(".panels svg path.ln").count() >= 2
        assert not errors

    def test_the_other_two_views_open(self, joined, open_page):
        page, errors = open_page(joined, cookie=joined.cookie)
        page.click("nav.tabs a:has-text('Fit')")
        page.wait_for_selector("#fit-views button")
        page.click("#fit-views button:has-text('What it cost to be right')")
        page.wait_for_selector("#fit-heading:has-text('What it cost to be right')")
        page.click("#fit-views button:has-text('What it has spent')")
        page.wait_for_selector("#fit-heading:has-text('What it has spent')")
        assert "answers no questions" in page.locator("#fit-body").inner_text()
        assert not errors

    def test_the_fit_page_on_its_own_carries_no_other_screen(self, open_page):
        """`ml-stack-serve fit --ui` puts up the same component with no daemon behind it."""
        import threading

        from ml_stack.fleet.ui import serve_page

        httpd = serve_page(name="atrium")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            served = type("Served", (), {"port": httpd.server_port})()
            page, errors = open_page(served, path="/ui/fit")
            page.wait_for_selector("table.fit tbody tr")
            assert page.locator("fleet-nav").count() == 0
            assert page.locator("cluster-view").count() == 0
            assert not errors
        finally:
            httpd.shutdown()
            httpd.server_close()
