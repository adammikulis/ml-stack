"""The rendered graph page, driven in headless Chromium.

Each test drives the page the way a person does — load it, click the buttons, select a
node, ask a question — and each one is paired with a one-line edit to graph.html that
makes it fail. The pairing is named in the test's docstring.

Everything skips when Playwright, Chromium, or the vendored libraries are unavailable.
The graph under test is invented; no test reads any data file.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")

from ml_stack.graph import page as graph_page  # noqa: E402

VENDOR = Path(__file__).resolve().parent / "support" / "vendor"

# the exact URLs graph.html loads, pinned so a CDN swap fails loudly
LIBS = {
    "https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js": (
        "d3.min.js",
        "f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539"),
    "https://cdnjs.cloudflare.com/ajax/libs/topojson/3.0.2/topojson.min.js": (
        "topojson.min.js",
        "b47a003c6a0d761211dbc60797d0d62f37917ddc228241fb38205732b1d78683"),
    "https://cdn.jsdelivr.net/npm/3d-force-graph@1.80.0/dist/3d-force-graph.min.js": (
        "3d-force-graph.min.js",
        "d96e738edcca580edd524730c1c6b05ed2efce028c23ca95db1bf43033a72e42"),
}

EMPTY_WORLD = {"type": "Topology",
               "objects": {"land": {"type": "GeometryCollection", "geometries": []}},
               "arcs": []}


def sample_graph():
    """Four nodes, two edges, one quoted message. Names from tests/known-fixtures.txt."""
    return {
        "nodes": [
            {"id": "person:ada", "label": "Ada Lovelace", "kind": "person", "mentions": 3,
             "attrs": {"member": True}, "messages": ["m1"]},
            {"id": "person:grace", "label": "Grace Hopper", "kind": "person", "mentions": 2,
             "attrs": {"member": True}, "messages": []},
            {"id": "org:quenlow", "label": "Quenlow Robotics", "kind": "org", "mentions": 1,
             "attrs": {}, "messages": []},
            {"id": "topic:iron", "label": "iron", "kind": "topic", "mentions": 2,
             "attrs": {}, "messages": ["m1"]},
        ],
        "edges": [
            {"source": "person:ada", "target": "topic:iron", "rel": "works_on",
             "weight": 2, "messages": ["m1"]},
            {"source": "person:grace", "target": "org:quenlow", "rel": "works_at",
             "weight": 1, "messages": []},
        ],
        # "environment" holds "iron" mid-word; only the standalone "iron" is a link
        "messages": {"m1": {"text": "We talked about iron all day; the environment came up too.",
                            "ts": "1700000000", "channel": "#general", "sender": "Ada Lovelace"}},
        "stats": {"messages": 1},
        "meta": {},
    }


def document(graph, *, served=False):
    """The rendered page as a browser-complete document."""
    body = graph_page.render(graph, title="A graph", world=EMPTY_WORLD)
    live = "<script>window.GRAPH_LIVE = true</script>" if served else ""
    return f'<!doctype html>\n<meta charset="utf-8">\n{live}{body}'


@pytest.fixture(scope="session")
def vendored():
    """The three libraries the page loads, on disk and matching their pins."""
    files = {}
    for url, (name, digest) in LIBS.items():
        path = VENDOR / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            VENDOR.mkdir(parents=True, exist_ok=True)
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    body = r.read()
            except OSError as exc:
                pytest.skip(f"cannot fetch {name}: {exc}")
            if hashlib.sha256(body).hexdigest() != digest:
                pytest.skip(f"{name} from the CDN does not match its pinned sha256")
            path.write_bytes(body)
        files[url] = path
    return files


@pytest.fixture(scope="session")
def browser():
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader",
                                        "--enable-unsafe-swiftshader"])
        except Exception as exc:
            pytest.skip(f"chromium did not launch: {exc}")
        yield b
        b.close()


@pytest.fixture()
def open_page(browser, vendored):
    """Opens the page in a fresh context; returns ``(page, errors)``."""
    contexts = []

    def _open(graph=None, *, view="2d", served=False, ask_reply=None):
        html = document(graph if graph is not None else sample_graph(), served=served)
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        contexts.append(ctx)
        ctx.add_init_script(
            f"try {{ localStorage.setItem('graph-view', '{view}') }} catch (e) {{}}")

        def route(r):
            url = r.request.url
            if url in vendored:
                r.fulfill(path=str(vendored[url]), content_type="application/javascript")
            elif url == "http://graph.test/":
                r.fulfill(body=html, content_type="text/html")
            elif url == "http://graph.test/ask" and r.request.method == "POST" \
                    and ask_reply is not None:
                r.fulfill(body=json.dumps(ask_reply), content_type="application/json")
            elif "fonts.googleapis.com" in url:
                r.fulfill(body="", content_type="text/css")
            else:
                r.abort()

        ctx.route("**/*", route)
        page = ctx.new_page()
        page.set_default_timeout(10_000)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text)
                if m.type == "error" and "Failed to load resource" not in m.text else None)
        page.goto("http://graph.test/")
        return page, errors

    yield _open
    for ctx in contexts:
        ctx.close()


def settle(page):
    page.wait_for_selector(".graph-wrap:not(.settling)")


def test_the_page_loads_clean(open_page):
    """The counts, the libraries, the closed panel, the placeholder — with no errors."""
    page, errors = open_page()
    page.wait_for_selector("#stats b")
    stats = page.locator("#stats span").all_inner_texts()
    assert "4 nodes" in stats
    assert "2 links" in stats
    assert page.evaluate("[typeof d3, typeof topojson, typeof ForceGraph3D]") \
        == ["object", "object", "function"]
    assert page.get_attribute("#display-btn", "aria-expanded") == "false"
    assert page.locator(".menu.display-open").count() == 0
    assert page.locator("#detail .placeholder h3").text_content() \
        == "What is here, and how it joins up"
    assert errors == []


def test_arrange_and_recenter_still_respond(open_page):
    """Fails when the two addEventListener lines for #arrange/#recenter are deleted."""
    page, errors = open_page()
    settle(page)
    root = "document.querySelector('#graph > g').getAttribute('transform')"
    box = page.locator("#graph").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    before = page.evaluate(root)
    page.mouse.wheel(0, -240)
    page.wait_for_function(f"v => {root} !== v", arg=before)
    zoomed = page.evaluate(root)
    page.click("#recenter")
    page.wait_for_function(f"v => {root} !== v", arg=zoomed)
    placed = "document.querySelector('#graph g.node').getAttribute('transform')"
    at = page.evaluate(placed)
    page.click("#arrange")
    page.wait_for_function(f"v => {placed} !== v", arg=at)
    assert errors == []


def test_a_modifier_click_in_3d_does_not_throw(open_page):
    """Fails when the capture-phase modifier-click loop on the wrap is deleted."""
    page, errors = open_page(view="3d")
    if not page.evaluate("!!document.createElement('canvas').getContext('webgl2')"):
        pytest.skip("no WebGL2 in this Chromium")
    try:
        page.wait_for_selector("#graph3d canvas")
    except pw.TimeoutError:
        pytest.skip("the 3D canvas never appeared")
    page.wait_for_selector("#labels3d span:not([hidden])", state="attached")
    box = page.locator(".graph-wrap").bounding_box()
    errors.clear()
    picked = page.locator("#picked button")
    deadline = time.monotonic() + 10
    page.keyboard.down("Shift")
    # a label sits just off its node, so clicks around each label cross the node itself
    while not errors and picked.count() == 0 and time.monotonic() < deadline:
        labels = page.eval_on_selector_all(
            "#labels3d span:not([hidden])",
            "els => els.map(e => ({x: parseFloat(e.style.left),"
            " y: parseFloat(e.style.top)}))")
        for lab in labels:
            for dy in range(-26, 27, 6):
                page.mouse.click(box["x"] + lab["x"], box["y"] + lab["y"] + dy)
                if errors or picked.count():
                    break
            else:
                continue
            break
    page.keyboard.up("Shift")
    assert errors == []
    assert picked.count() > 0


def test_quote_links_respect_word_boundaries(open_page):
    """Fails when the linkify RegExp flags go from 'giu' to 'gi'."""
    page, errors = open_page()
    settle(page)
    page.locator("#graph g.node", has_text="Ada Lovelace").first.press("Enter")
    page.wait_for_selector("#detail blockquote.msg")
    assert page.locator('#detail button.ref[data-id="topic:iron"]').count() == 1
    assert errors == []


def test_an_answer_counts_only_the_nodes_it_names(open_page):
    """Fails when the u flag is dropped from the word-splitting regex in words()."""
    reply = {"content": "Ada Lovelace kept coming back to it.",
             "ids": ["person:ada", "topic:iron", "org:quenlow"], "why": ""}
    page, errors = open_page(served=True, ask_reply=reply)
    page.wait_for_selector("#stats b")
    page.fill("#q", "who kept coming back to iron?")
    page.press("#q", "Enter")
    pw.expect(page.locator("#detail h3")).to_have_text("In this answer · 3")
    assert "The first 1" in page.locator("#detail p.aside-note").inner_text()
    assert errors == []


def test_hidden_beats_any_display_rule(open_page):
    """Fails when .detail gains a display rule and .detail[hidden] loses its own."""
    page, errors = open_page()
    page.wait_for_selector("#detail")
    shown = page.eval_on_selector(
        "#detail",
        "el => { el.hidden = true; const d = getComputedStyle(el).display;"
        " el.hidden = false; return d; }")
    assert shown == "none"
    assert errors == []


def test_a_second_ask_carries_what_is_lit(open_page):
    """Fails when the held array is dropped from the /ask POST body."""
    reply = {"content": "Ada Lovelace works on iron.",
             "ids": ["person:ada", "topic:iron"], "why": ""}
    page, errors = open_page(served=True, ask_reply=reply)
    page.wait_for_selector("#stats b")
    page.fill("#q", "who works on iron?")
    with page.expect_request("**/ask") as first:
        page.press("#q", "Enter")
    pw.expect(page.locator("#detail h3")).to_have_text("In this answer \u00b7 2")
    page.fill("#q", "and the org?")
    with page.expect_request("**/ask") as second:
        page.press("#q", "Enter")
    assert "held" not in json.loads(first.value.post_data)
    assert json.loads(second.value.post_data)["held"] == ["person:ada", "topic:iron"]
    assert errors == []
