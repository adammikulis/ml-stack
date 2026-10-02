"""ml-ui: the verdict definition, the tokens' contrast, the served assets, and every element
driven in headless Chromium through the gallery page the daemon serves."""

from __future__ import annotations

import re

import pytest

from ml_stack.ui import assets, assets_dir
from ml_stack.ui.verdict import LABELS, THRESHOLDS, verdict_of


# -- the verdict definition -----------------------------------------------------------------
@pytest.mark.parametrize("used,capacity,expected", [
    (0, 10, "green"), (7.9, 10, "green"), (8, 10, "yellow"), (9.4, 10, "yellow"),
    (9.5, 10, "red"), (30, 10, "red"), (1, 0, "none"), (-1, 10, "none"),
])
def test_a_share_of_capacity_reads_green_yellow_or_red(used, capacity, expected):
    assert verdict_of(used, capacity) == expected


def test_every_verdict_has_a_word():
    assert set(LABELS) == {"green", "yellow", "red", "none"}
    assert THRESHOLDS.yellow_at < THRESHOLDS.red_at


def test_the_folder_holds_what_the_page_loads():
    names = set(assets())
    assert {"ml-ui.css", "ml-ui.js", "gallery.html", "verdict.json"} <= names
    html = (assets_dir() / "ml-ui.js").read_text(encoding="utf-8")
    for module in re.findall(r'from "\./([\w.-]+)"|import "\./([\w.-]+)"', html):
        assert (module[0] or module[1]) in names


# -- the tokens ----------------------------------------------------------------------------
def luminance(hexcolour):
    channels = [int(hexcolour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def blend(fill, over, share):
    mixed = [round(int(fill[i:i + 2], 16) * share + int(over[i:i + 2], 16) * (1 - share))
             for i in (1, 3, 5)]
    return "#" + "".join(f"{c:02x}" for c in mixed)


def tokens():
    css = (assets_dir() / "ml-ui.css").read_text(encoding="utf-8")
    dark = dict(re.findall(r"--ml-([\w-]+):\s*(#[0-9a-f]{6})", css.split("@media")[0]))
    media = css.split("@media (prefers-color-scheme: light)")[1].split('}\n}')[0]
    light = dict(re.findall(r"--ml-([\w-]+):\s*(#[0-9a-f]{6})", media))
    forced = css.split(':root[data-theme="light"]')[1].split("}")[0]
    assert dict(re.findall(r"--ml-([\w-]+):\s*(#[0-9a-f]{6})", forced)) == light
    return {"dark": dark, "light": {**dark, **light}}


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_text_and_verdicts_meet_wcag_aa_in_both_themes(theme):
    t = tokens()[theme]
    for surface in ("bg", "surface", "sunken"):
        assert contrast(t["text"], t[surface]) >= 4.5
        assert contrast(t["muted"], t[surface]) >= 4.5
    for verdict in ("green", "yellow", "red"):
        for surface in ("surface", "sunken"):
            chip = blend(t[verdict], t[surface], 0.16)
            assert contrast(t[f"{verdict}-ink"], chip) >= 4.5, (verdict, surface)
            assert contrast(t[verdict], t[surface]) >= 3, (verdict, surface)
    assert contrast(t["line-strong"], t["surface"]) >= 3
    assert contrast(t["accent-ink"], t["surface"]) >= 4.5
    assert contrast(t["on-accent"], t["accent"]) >= 4.5


# -- the assets, served ----------------------------------------------------------------------
from test_fleet_ui import Serving  # noqa: E402


@pytest.fixture
def daemon(tmp_path):
    served = Serving(tmp_path)
    try:
        yield served
    finally:
        served.close()


@pytest.fixture(scope="session")
def browser(request):
    pytest.importorskip("playwright.sync_api", reason="ml-stack[scrape]")
    try:
        b = request.getfixturevalue("playwright").chromium.launch(headless=True)
    except Exception as exc:                           # noqa: BLE001
        pytest.skip(f"chromium did not launch: {exc}")
    yield b
    b.close()


@pytest.fixture
def open_page(browser):
    contexts = []

    def _open(served, *, path):
        ctx = browser.new_context(viewport={"width": 1100, "height": 900})
        contexts.append(ctx)
        page = ctx.new_page()
        page.set_default_timeout(10_000)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(f"http://127.0.0.1:{served.port}{path}")
        return page, errors

    yield _open
    for ctx in contexts:
        ctx.close()


def test_the_daemon_serves_the_folder_flat_and_nothing_else(daemon):
    status, _, headers = daemon.call("/ui/ml-ui/ml-ui.css")
    assert status == 200 and "text/css" in headers["Content-Type"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    status, _, headers = daemon.call("/ui/ml-ui/ml-ui.js")
    assert status == 200 and "javascript" in headers["Content-Type"]
    assert daemon.call("/ui/ml-ui/../daemon.py")[0] == 404
    assert daemon.call("/ui/ml-ui/nothing.js")[0] == 404
    assert daemon.call("/ui/gallery")[0] == 302


@pytest.fixture
def gallery(daemon, open_page):
    def _open(theme="dark"):
        page, errors = open_page(daemon, path=f"/ui/ml-ui/gallery.html?theme={theme}")
        page.wait_for_selector("ml-meter")
        page.wait_for_timeout(300)
        return page, errors
    return _open


@pytest.mark.slow
class TestGallery:
    def test_every_state_draws_without_an_error_and_runs_nothing_it_was_given(self, gallery):
        page, errors = gallery()
        assert errors == []
        assert page.evaluate("window.__pwned") is None
        for tag in ("ml-meter", "ml-chip", "ml-progress", "ml-sparkline", "ml-table",
                    "ml-stepper", "ml-select", "ml-slider", "ml-toggle", "ml-path"):
            assert page.locator(tag).count() > 0, tag

    def test_hostile_text_is_shown_as_text_and_never_becomes_markup(self, gallery):
        page, _ = gallery()
        found = page.evaluate("""() => {
          const hostile = [...document.querySelectorAll('*')].flatMap(
            (n) => [n, ...(n.shadowRoot ? n.shadowRoot.querySelectorAll('*') : [])]);
          return {
            images: hostile.filter((n) => n.tagName === 'IMG').length,
            shown: [...document.querySelectorAll('ml-meter')].some((m) =>
              m.shadowRoot.querySelector('.label').textContent.includes('<img src=x')),
          };
        }""")
        assert found == {"images": 0, "shown": True}

    def test_a_meter_names_its_verdict_in_words_and_in_the_aria_value(self, gallery):
        page, _ = gallery()
        red = page.evaluate("""() => {
          const m = [...document.querySelectorAll('ml-meter')].find(
            (x) => x.getAttribute('label') === 'Video memory' && x.capacity && x.segments[0].value === 20e9
              && x.segments[1].value === 9e9);
          const track = m.shadowRoot.querySelector('[role=meter]');
          const chip = m.shadowRoot.querySelector('ml-chip').shadowRoot;
          return { text: track.getAttribute('aria-valuetext'), now: track.getAttribute('aria-valuenow'),
                   word: chip.textContent, icon: !!chip.querySelector('svg') };
        }""")
        assert "Over" in red["text"] and "Over" in red["word"] and red["icon"]
        assert float(red["now"]) == 24e9

    def test_the_verdict_json_is_the_definition_the_script_holds(self, gallery):
        page, _ = gallery()
        got = page.evaluate("""async () => {
          const v = await import('./verdict.js');
          const j = await (await fetch('verdict.json')).json();
          return [v.THRESHOLDS, j, v.LABELS];
        }""")
        js, definition, labels = got
        assert js == {"yellowAt": definition["yellow_at"], "redAt": definition["red_at"]}
        assert labels == definition["labels"] == LABELS
        assert js["yellowAt"] == THRESHOLDS.yellow_at and js["redAt"] == THRESHOLDS.red_at

    def test_a_property_set_before_the_element_exists_is_kept(self, gallery):
        page, _ = gallery()
        text = page.evaluate("""() => {
          const other = document.implementation.createHTMLDocument('');
          const m = other.createElement('ml-meter');
          m.segments = [{ label: 'early', value: 3 }];
          m.capacity = 10;
          document.body.append(document.adoptNode(m));
          return new Promise((r) => setTimeout(() =>
            r(m.shadowRoot.querySelector('[role=meter]').getAttribute('aria-valuetext')), 50));
        }""")
        assert "3.0 of 10.0" in text and "OK" in text

    def test_progress_cancel_fires_an_event_and_indeterminate_has_no_value(self, gallery):
        page, _ = gallery()
        got = page.evaluate("""() => {
          const [bar, loose] = [...document.querySelectorAll('ml-progress')];
          let n = 0;
          bar.addEventListener('ml-cancel', () => { n += 1; });
          bar.shadowRoot.querySelector('button').click();
          return [n, bar.shadowRoot.querySelector('[role=progressbar]').getAttribute('aria-valuenow'),
                  loose.shadowRoot.querySelector('[role=progressbar]').hasAttribute('aria-valuenow')];
        }""")
        assert got == [1, "35", False]

    def test_the_stepper_refuses_a_step_that_fails_validation_and_moves_when_it_passes(
            self, gallery):
        page, _ = gallery()
        got = page.evaluate("""async () => {
          const wait = () => new Promise((r) => setTimeout(r, 30));
          const bad = document.getElementById('st2'), good = document.getElementById('st1');
          bad.shadowRoot.querySelector('.primary').click();
          await wait();
          const alert = bad.shadowRoot.querySelector('[role=alert]');
          const seen = [];
          good.addEventListener('ml-step', (e) => seen.push(e.detail.to));
          good.shadowRoot.querySelector('.primary').click();
          await wait();
          return { refused: !alert.hidden && alert.textContent, stayed: bad.current,
                   moved: good.current, seen,
                   mark: good.shadowRoot.querySelector('[aria-current=step]').textContent };
        }""")
        assert got["refused"] == "Pick at least one job first."
        assert got["stayed"] == 1 and got["moved"] == 1 and got["seen"] == [1]
        assert got["mark"] == "Cluster"

    def test_the_stepper_state_resumes_where_it_was(self, gallery):
        page, _ = gallery()
        got = page.evaluate("""() => {
          const s = document.getElementById('st1');
          s.state = { current: 2, completed: [0, 1] };
          return new Promise((r) => setTimeout(() => r([s.state,
            s.shadowRoot.querySelector('slot').getAttribute('name')]), 30));
        }""")
        assert got == [{"current": 2, "completed": [0, 1]}, "step-job"]

    def test_the_sheet_traps_focus_closes_on_escape_and_hands_focus_back(self, gallery):
        page, _ = gallery()
        opener = page.get_by_text("Open sheet")
        opener.focus()
        opener.click()
        page.wait_for_timeout(100)
        assert page.evaluate("document.getElementById('sheet').shadowRoot.querySelector('dialog').open")
        for _ in range(6):
            page.keyboard.press("Tab")
        inside = page.evaluate("""() => document.getElementById('sheet').contains(document.activeElement)
            || document.getElementById('sheet').shadowRoot.contains(
                 document.getElementById('sheet').shadowRoot.activeElement)""")
        assert inside
        page.keyboard.press("Escape")
        page.wait_for_timeout(100)
        assert not page.evaluate("document.getElementById('sheet').hasAttribute('open')")
        assert page.evaluate("document.activeElement.textContent") == "Open sheet"

    def test_a_table_sorts_by_a_column_and_reports_the_chosen_row(self, gallery):
        page, _ = gallery()
        got = page.evaluate("""async () => {
          const t = document.querySelector('ml-table[selectable]');
          const names = () => [...t.shadowRoot.querySelectorAll('tbody tr .name')].map((n) => n.textContent);
          t.shadowRoot.querySelectorAll('th button')[2].click();
          await new Promise((r) => setTimeout(r, 30));
          const first = names()[0];
          let picked = null;
          t.addEventListener('ml-row', (e) => { picked = e.detail.id; });
          t.shadowRoot.querySelector('tbody tr').click();
          return [first, picked, t.sortKey, t.sortDir];
        }""")
        assert got[0].startswith("<img")
        assert got[2] == "size"
        assert got[1] in {"a", "b", "c", "d"}

    def test_a_slider_over_stops_reports_the_stop_not_the_index(self, gallery):
        page, _ = gallery()
        got = page.evaluate("""async () => {
          const s = document.querySelector('ml-slider');
          const seen = [];
          s.addEventListener('change', (e) => seen.push(e.detail.value));
          const input = s.shadowRoot.querySelector('input');
          input.value = '6';
          input.dispatchEvent(new Event('input'));
          await new Promise((r) => setTimeout(r, 30));
          return [seen, s.value, s.shadowRoot.querySelector('output').textContent];
        }""")
        assert got[0] == [131072] and got[1] == 131072
        assert got[2].startswith("131072")

    def test_a_toggle_flips_on_a_click_and_says_which_it_is(self, gallery):
        page, _ = gallery()
        got = page.evaluate("""async () => {
          const t = document.querySelector('ml-toggle');
          const b = t.shadowRoot.querySelector('[role=switch]');
          const before = b.getAttribute('aria-checked');
          b.click();
          await new Promise((r) => setTimeout(r, 30));
          return [before, t.checked, b.getAttribute('aria-checked')];
        }""")
        assert got == ["true", False, "false"]

    def test_a_toast_is_announced_and_goes_away_on_dismiss(self, gallery):
        page, _ = gallery()
        page.get_by_text("red toast").click()
        toaster = page.locator("#toaster")
        assert toaster.evaluate("t => t.shadowRoot.querySelector('[role=alert] .toast') !== null")
        toaster.evaluate("t => t.shadowRoot.querySelector('.x').click()")
        assert toaster.evaluate("t => t.shadowRoot.querySelectorAll('.toast').length") == 0

    @pytest.mark.parametrize("theme", ["dark", "light"])
    def test_both_themes_paint_the_page_in_their_own_surface(self, gallery, theme):
        page, _ = gallery(theme)
        bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
        assert bg == ("rgb(11, 15, 20)" if theme == "dark" else "rgb(246, 248, 250)")

    def test_reduced_motion_stops_the_indeterminate_slide(self, gallery):
        page, _ = gallery()
        page.emulate_media(reduced_motion="reduce")
        name = page.evaluate("""() => getComputedStyle(
          document.querySelectorAll('ml-progress')[1].shadowRoot.querySelector('.fill')).animationName""")
        assert name == "none"
