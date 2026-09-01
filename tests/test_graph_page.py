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
    """Four nodes, two edges, two quoted messages a day apart.

    Names from tests/known-fixtures.txt.
    """
    return {
        "nodes": [
            {"id": "person:ada", "label": "Ada Lovelace", "kind": "person", "mentions": 3,
             "attrs": {"member": True}, "messages": ["m1"]},
            {"id": "person:grace", "label": "Grace Hopper", "kind": "person", "mentions": 2,
             "attrs": {"member": True}, "messages": ["m2"]},
            {"id": "org:quenlow", "label": "Quenlow Robotics", "kind": "org", "mentions": 1,
             "attrs": {}, "messages": ["m2"]},
            {"id": "topic:iron", "label": "iron", "kind": "topic", "mentions": 2,
             "attrs": {}, "messages": ["m1"]},
        ],
        "edges": [
            {"source": "person:ada", "target": "topic:iron", "rel": "works_on",
             "weight": 2, "messages": ["m1"]},
            {"source": "person:grace", "target": "org:quenlow", "rel": "works_at",
             "weight": 1, "messages": ["m2"]},
        ],
        # "environment" holds "iron" mid-word; only the standalone "iron" is a link
        "messages": {"m1": {"text": "We talked about iron all day; the environment came up too.",
                            "ts": "1700000000", "channel": "#general", "sender": "Ada Lovelace"},
                     "m2": {"text": "I started at Quenlow Robotics this week.",
                            "ts": "1700086400", "channel": "#general", "sender": "Grace Hopper"}},
        "stats": {"messages": 2},
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
            # headless is the default, said out loud: a test must never take the screen
            b = p.chromium.launch(headless=True,
                                  args=["--use-gl=angle", "--use-angle=swiftshader",
                                        "--enable-unsafe-swiftshader"])
        except Exception as exc:
            pytest.skip(f"chromium did not launch: {exc}")
        yield b
        b.close()


@pytest.fixture()
def open_page(browser, vendored):
    """Opens the page in a fresh context; returns ``(page, errors)``."""
    contexts = []

    def _open(graph=None, *, view="2d", served=False, ask_reply=None, ask_stream=None,
              review=None, origin="http://graph.test/"):
        html = document(graph if graph is not None else sample_graph(), served=served)
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        contexts.append(ctx)
        ctx.add_init_script(
            f"try {{ localStorage.setItem('graph-view', '{view}') }} catch (e) {{}}")

        def route(r):
            url = r.request.url
            if url in vendored:
                r.fulfill(path=str(vendored[url]), content_type="application/javascript")
            elif url == origin:
                r.fulfill(body=html, content_type="text/html")
            elif url == origin + "ask" and r.request.method == "POST" \
                    and ask_reply is not None:
                r.fulfill(body=json.dumps(ask_reply), content_type="application/json")
            elif url == origin + "ask/stream" and r.request.method == "POST" \
                    and ask_stream is not None:
                r.fulfill(body=ask_stream, content_type="text/event-stream")
            elif url == origin + "review" and review is not None:
                r.fulfill(body=json.dumps({"ok": True, "problems": []}
                                          if r.request.method == "POST" else review),
                          content_type="application/json")
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
        page.goto(origin)
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


def test_an_answer_lights_only_what_it_names(open_page):
    """Fails when the u flag is dropped from the word-splitting regex in words().

    A model that does not call ``show`` leaves the prose as the only witness to what its
    answer was about. Lighting everything the tools touched instead lit the topic a question
    about people was found through, and left the people dark.
    """
    reply = {"content": "Ada Lovelace kept coming back to it.",
             "ids": ["person:ada", "topic:iron", "org:quenlow"], "why": ""}
    page, errors = open_page(served=True, ask_reply=reply)
    page.wait_for_selector("#stats b")
    page.fill("#q", "who kept coming back to iron?")
    page.press("#q", "Enter")
    pw.expect(page.locator("#detail h3")).to_have_text("In this answer · 1")
    assert page.locator('#detail .links button[data-id="person:ada"]').count() == 1
    assert page.locator('#detail .links button[data-id="org:quenlow"]').count() == 0
    assert errors == []


def test_show_beats_everything_else_the_tools_touched(open_page):
    """What lights up is what the model says its answer is about, not its working."""
    reply = {"content": "They both keep at it.",
             "ids": ["topic:iron"], "read": ["topic:iron"],
             "show": ["person:ada", "org:quenlow"], "why": ""}
    page, errors = open_page(served=True, ask_reply=reply)
    page.wait_for_selector("#stats b")
    page.fill("#q", "who keeps at iron?")
    page.press("#q", "Enter")
    pw.expect(page.locator("#detail h3")).to_have_text("In this answer · 2")
    assert page.locator('#detail .links button[data-id="person:ada"]').count() == 1
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


def test_a_second_ask_carries_only_what_was_gathered(open_page):
    """Fails when the held array is dropped, or when it is filled from the last answer.

    An answer's own nodes are not a standing selection: carrying them into the next
    question drags its strangers along. Only a deliberate pick or selection travels.
    """
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
    # the answer lit two nodes, and neither was gathered by hand
    assert "held" not in json.loads(second.value.post_data)
    # gathering one deliberately does travel
    page.locator("#graph g.node", has_text="Ada Lovelace").first.click(modifiers=["Shift"])
    page.fill("#q", "what about them?")
    with page.expect_request("**/ask") as third:
        page.press("#q", "Enter")
    assert json.loads(third.value.post_data)["held"] == ["person:ada"]
    assert errors == []


def test_history_replays_the_messages_in_order(open_page):
    """Fails when the chronological sort in buildHistory runs newest-first."""
    page, errors = open_page()
    settle(page)
    assert page.locator(".tools #history").count() == 1
    page.evaluate("window.__historyClock = 3000")
    page.click("#history")
    assert page.text_content("#history") == "■ stop"
    assert page.get_attribute("#history", "aria-pressed") == "true"
    # the first pulse lights exactly one edge: the older message's — works_on, index 0
    first = page.wait_for_function(
        "() => { const lit = [...document.querySelectorAll('#graph .link')]"
        ".flatMap((l, i) => (l.classList.contains('lit') ? [i] : []));"
        " return lit.length === 1 ? lit[0] + 1 : false; }"
    ).json_value() - 1
    assert first == 0
    page.wait_for_selector("#graph circle.orb-core", state="attached")
    page.wait_for_function(
        "() => document.querySelectorAll('#graph .node.lit').length === 2")
    caption = page.text_content("#history-when")
    assert caption and "Ada" not in caption and "iron" not in caption
    page.click("#history")
    assert page.locator("#graph circle.orb-core, #graph circle.orb-halo").count() == 0
    assert page.locator("#graph .link.lit").count() == 0
    assert page.locator("#graph .node.lit").count() == 0
    assert page.text_content("#history") == "▶ history"
    assert page.get_attribute("#history", "aria-pressed") == "false"
    assert page.is_hidden("#history-when")
    assert errors == []


def test_a_finished_history_run_leaves_nothing_behind(open_page):
    """Fails when the closing stopHistory timer is dropped from playHistory."""
    page, errors = open_page()
    settle(page)
    page.evaluate("window.__historyClock = 400")
    page.click("#history")
    assert page.get_attribute("#history", "aria-pressed") == "true"
    page.wait_for_function(
        "() => document.getElementById('history').getAttribute('aria-pressed') === 'false'")
    assert page.locator("#graph circle.orb-core, #graph circle.orb-halo").count() == 0
    assert page.locator("#graph .link.lit").count() == 0
    assert page.locator("#graph .node.lit").count() == 0
    assert page.text_content("#history") == "▶ history"
    assert page.is_hidden("#history-when")
    assert errors == []


REVIEW = [
    {"id": "r1", "index": 1, "at": "2026-01-02T09:00:00Z", "kind": "Fix my information",
     "attested": True, "claimed": "person:ada", "claimedLabel": "Ada Lovelace",
     "concerns": [], "text": "I do not work on iron any more",
     "targets": [{"key": "node:topic:iron", "label": "iron"}],
     "edits": [{"op": "remove_edge", "target": "person:ada", "other": "topic:iron",
                "name": "works_on", "reason": "asked to", "problems": []}],
     "status": "proposed"},
    {"id": "r2", "index": 2, "at": "2026-01-03T09:00:00Z", "kind": "Remove something",
     "attested": False, "claimed": "", "claimedLabel": "Grace Hopper",
     "concerns": ["not attested as their own information"],
     "text": "take that org out", "targets": [],
     "edits": [{"op": "remove_node", "target": "org:quenlow", "reason": "asked",
                "problems": ["the entry 'org:quenlow' is not in the graph"]}],
     "status": "accepted"},
]


def test_the_review_panel_lists_what_the_server_sent(open_page):
    """Fails when the loadReview() call at the bottom of the review block is deleted."""
    page, errors = open_page(served=True, review=REVIEW, origin="http://127.0.0.1/")
    page.wait_for_selector("#review-box:not([hidden])", state="attached")
    page.eval_on_selector("#ask-box", "el => { el.open = true }")
    # an addressed request has left the list; the count is what still waits
    assert page.text_content("#review-count") == "1"
    page.eval_on_selector("#review-box", "el => { el.open = true }")
    reqs = page.locator("#review-list .req")
    assert reqs.count() == 1
    first = reqs.nth(0).inner_text()
    assert "2026-01-02 09:00" in first and "Fix my information" in first
    assert "attested" in first and "NOT ATTESTED" not in first
    assert "from: Ada Lovelace" in first
    assert "unjoin person:ada -works_on-> topic:iron — asked to" in first
    assert reqs.nth(0).locator('button[data-act="accept"]').count() == 1
    assert reqs.nth(0).locator('button[data-act="refuse"]').count() == 1
    # the addressed one is a toggle away, so Undo stays reachable
    toggle = page.locator("#review-done")
    assert "1 addressed" in toggle.inner_text()
    toggle.click()
    reqs = page.locator("#review-list .req")
    assert reqs.count() == 2
    second = reqs.nth(1).inner_text()
    assert "NOT ATTESTED" in second
    assert "not attested as their own information" in second
    assert "the graph says: the entry 'org:quenlow' is not in the graph" in second
    assert reqs.nth(1).locator('button[data-act="undo"]').count() == 1
    assert "Refresh graph" in page.text_content("#review-box .review-note")
    assert errors == []


def test_accepting_posts_the_id_and_refetches_the_list(open_page):
    """Fails when the POST body in paintReview loses its action field."""
    page, errors = open_page(served=True, review=REVIEW, origin="http://127.0.0.1/")
    page.wait_for_selector("#review-box:not([hidden])", state="attached")
    page.eval_on_selector("#ask-box", "el => { el.open = true }")
    page.eval_on_selector("#review-box", "el => { el.open = true }")
    with page.expect_request(
            lambda r: r.url == "http://127.0.0.1/review" and r.method == "POST") as posted, \
         page.expect_request(
            lambda r: r.url == "http://127.0.0.1/review" and r.method == "GET") as refetched:
        page.click('#review-list button[data-act="accept"]')
    assert json.loads(posted.value.post_data) == {"id": "r1", "action": "accept"}
    assert refetched.value is not None
    page.wait_for_selector("#review-list .req")
    assert errors == []


def test_a_page_not_on_loopback_never_shows_review(open_page):
    """Fails when the review block stops standing behind the loopback gate."""
    page, errors = open_page(served=True, review=REVIEW)
    page.wait_for_selector("#stats b")
    page.wait_for_timeout(500)
    assert page.locator("#review-box").is_hidden()
    assert page.locator("#review-list .req").count() == 0
    assert errors == []


def test_what_was_read_lights_up_not_everything_found(open_page):
    """A broad search must not flood the graph: read+path light, found alone is a fallback."""
    reply = {"content": "Ada works on iron.", "ids": ["person:ada", "topic:iron", "org:quenlow"],
             "read": ["person:ada"], "path": [], "found": ["topic:iron", "org:quenlow"], "why": ""}
    page, errors = open_page(served=True, ask_reply=reply)
    page.wait_for_selector("#stats b")
    page.fill("#q", "who works on iron?")
    page.press("#q", "Enter")
    pw.expect(page.locator("#detail h3")).to_have_text("In this answer · 1")
    assert page.input_value("#q") == ""
    assert errors == []


def test_a_streamed_answer_fills_the_thinking_then_the_bubble(open_page):
    """Fails when the SSE loop in askStream stops feeding events into the bubble."""
    events = [
        {"event": "thinking", "text": "find who works iron. "},
        {"event": "tool", "name": "look_up", "detail": "'iron'"},
        {"event": "tool_result", "name": "look_up", "count": 2},
        {"event": "answer", "text": "Ada Lovelace "},
        {"event": "answer", "text": "works on iron."},
        {"event": "done", "content": "Ada Lovelace works on iron.",
         "ids": ["person:ada", "topic:iron"], "read": ["person:ada"], "path": [],
         "found": ["topic:iron"], "why": "looked up 'iron'"},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    page, errors = open_page(served=True, ask_stream=body)
    page.wait_for_selector("#stats b")
    page.fill("#q", "who works on iron?")
    page.press("#q", "Enter")
    # the answer names Ada and iron, and both are what it is about; before this it lit
    # whatever look_at had been given, which is the working rather than the answer
    pw.expect(page.locator("#detail h3")).to_have_text("In this answer · 2")
    trace = page.locator("#qturns .t .think .trace").text_content()
    assert "find who works iron." in trace
    assert "look_up 'iron'" in trace and "2 back" in trace
    assert page.locator("#qturns .t .said").inner_text() == "Ada Lovelace works on iron."
    # the trace folds away once the answer has landed
    assert page.get_attribute("#qturns .t .think", "open") is None
    assert "2 lit up" in page.text_content("#qnote")
    assert errors == []


def test_gathering_a_node_looks_like_something_in_3d(open_page):
    """Fails when picked nodes are not accent-coloured in the three-dimensional view.

    The 2D view outlines a gathered node; before this the same click in 3D changed
    nothing anyone could see, and the feature read as broken.
    """
    page, errors = open_page(view="3d")
    page.wait_for_selector("#graph3d canvas", state="attached")
    if not page.evaluate("!!document.createElement('canvas').getContext('webgl2')"):
        pytest.skip("no WebGL in this chromium")
    page.wait_for_selector("#labels3d span:not([hidden])")
    label = page.locator("#labels3d span:not([hidden])").first
    box = label.bounding_box()
    page.keyboard.down("Shift")
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] - 8)
    page.keyboard.up("Shift")
    # the colour on screen, not a class name: a class proves nothing a person can see.
    # A probe painted with the token gives the same rgb() string the browser computes
    accent = page.evaluate(
        "() => { const p = document.createElement('i'); p.style.color = 'var(--accent)';"
        " document.body.appendChild(p); const c = getComputedStyle(p).color;"
        " p.remove(); return c; }")
    page.wait_for_function(
        "a => [...document.querySelectorAll('#labels3d span')]"
        ".some(e => getComputedStyle(e).color === a)", arg=accent)
    lit = page.evaluate(
        "a => [...document.querySelectorAll('#labels3d span')]"
        ".filter(e => getComputedStyle(e).color === a).length", arg=accent)
    assert lit == 1
    # and the count is readable without opening anything: the panel is an overlay and
    # opening it would take the clicks meant for the graph, so the graph's hint line says it
    assert "1 gathered" in page.text_content(".tools .hint")
    assert errors == []


def test_a_plain_click_drops_what_was_gathered(open_page):
    """Shift gathers; a plain click starts again, and still selects what it hit."""
    page, errors = open_page()
    settle(page)
    nodes = page.locator("#graph g.node")
    nodes.nth(0).click(modifiers=["Shift"])
    nodes.nth(1).click(modifiers=["Shift"])
    assert page.locator("#graph g.node.picked").count() == 2
    nodes.nth(2).click()
    assert page.locator("#graph g.node.picked").count() == 0
    assert page.locator("#graph g.node.on").count() == 1
    assert errors == []


def test_gathering_a_second_node_in_3d_keeps_the_labels_moving(open_page):
    """Fails when a label freezes on screen while the graph turns under it.

    Gathering two nodes and turning the view left one name pinned to a screen position, and
    find paths did nothing — both of which are what a dead label loop looks like: the render
    loop keeps drawing spheres, and the names, which are DOM, stop where they were.
    """
    page, errors = open_page(view="3d")
    page.wait_for_selector("#graph3d canvas", state="attached")
    if not page.evaluate("!!document.createElement('canvas').getContext('webgl2')"):
        pytest.skip("no WebGL in this chromium")
    page.wait_for_selector("#labels3d span:not([hidden])")
    for i in range(2):
        label = page.locator("#labels3d span:not([hidden])").nth(i)
        box = label.bounding_box()
        page.keyboard.down("Shift")
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] - 8)
        page.keyboard.up("Shift")
    # the loop is alive if the placement pass keeps running: it stamps a frame counter
    page.evaluate("() => { window.__frames = 0; }")
    before = page.evaluate("() => [...document.querySelectorAll('#labels3d span')]"
                           ".map(e => e.style.left + ',' + e.style.top).join('|')")
    # turn the view, the way a reader would
    box = page.locator("#graph3d").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2 + 220, box["y"] + box["height"] / 2 + 60,
                    steps=12)
    page.mouse.up()
    page.wait_for_timeout(600)
    after = page.evaluate("() => [...document.querySelectorAll('#labels3d span')]"
                          ".map(e => e.style.left + ',' + e.style.top).join('|')")
    assert after != before, "the labels never moved: the placement loop is dead"
    assert errors == []


def test_find_paths_joins_what_was_gathered_in_3d(open_page):
    """Fails when find paths lights its nodes and draws no route between them."""
    page, errors = open_page(view="3d")
    page.wait_for_selector("#graph3d canvas", state="attached")
    if not page.evaluate("!!document.createElement('canvas').getContext('webgl2')"):
        pytest.skip("no WebGL in this chromium")
    page.wait_for_selector("#labels3d span:not([hidden])")
    # two named nodes, gathered by their own names: the list re-sorts after every pick, so
    # clicking "the first" twice clicks one node twice and toggles it straight back off
    names = page.evaluate("() => [...document.querySelectorAll('#labels3d span')]"
                          ".filter(e => !e.hidden && !e.className.includes('edge'))"
                          ".slice(0, 2).map(e => e.textContent)")
    assert len(names) == 2
    picked = page.locator("#picked button")
    for name in names:
        want = picked.count() + 1
        label = page.locator("#labels3d span", has_text=name).first
        box = label.bounding_box()
        # a name sits above its node, below it, or on it, and only the page knows which, so
        # the sphere is hunted for around the name rather than assumed to be one way up
        for dy in (-8, 14, 6, -20, 22, -30):
            page.keyboard.down("Shift")
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + dy)
            page.keyboard.up("Shift")
            if picked.count() == want:
                break
    assert picked.count() == 2, "a second shift-click gathered nothing"
    page.click("#findpath")
    page.wait_for_timeout(300)
    assert "joined" in page.text_content("#qcount") or page.text_content("#qcount").isdigit()
    assert errors == []

def joined_only_through_a_topic():
    """Two members whose only connection is a subject they share, and one direct pair.

    The shared fixture's two people are in different components, so no route exists either
    way and a test over it cannot tell the two modes apart.
    """
    return {
        "nodes": [
            {"id": "person:ada", "label": "Ada Lovelace", "kind": "person", "mentions": 3,
             "attrs": {"member": True}, "messages": []},
            {"id": "person:grace", "label": "Grace Hopper", "kind": "person", "mentions": 2,
             "attrs": {"member": True}, "messages": []},
            {"id": "person:alan", "label": "Alan Turing", "kind": "person", "mentions": 2,
             "attrs": {"member": True}, "messages": []},
            {"id": "topic:iron", "label": "iron", "kind": "topic", "mentions": 2,
             "attrs": {}, "messages": []},
        ],
        "edges": [
            {"source": "person:ada", "target": "topic:iron", "rel": "works_on", "weight": 2,
             "messages": []},
            {"source": "person:grace", "target": "topic:iron", "rel": "works_on", "weight": 2,
             "messages": []},
            {"source": "person:ada", "target": "person:alan", "rel": "works_with", "weight": 1,
             "messages": []},
        ],
        "messages": {},
        "stats": {"messages": 0},
        "meta": {},
    }


def gather(page, *ids):
    for node_id in ids:
        page.evaluate(
            """id => { const g = [...document.querySelectorAll('#graph g.node')]
                        .find(x => x.__data__.id === id);
                       const r = g.querySelector('circle.hit').getBoundingClientRect();
                       g.dispatchEvent(new MouseEvent('click', { bubbles: true, shiftKey: true,
                         clientX: r.x + r.width / 2, clientY: r.y + r.height / 2 })); }""",
            node_id)


def lit_kinds(page):
    return page.evaluate(
        "() => [...document.querySelectorAll('#graph g.node.answer')]"
        ".map(g => g.className.baseVal.match(/k-(\\w+)/)[1]).sort()")


def test_join_like_with_like_narrows_what_a_route_passes_through(open_page):
    """Fails when the chip does not actually change what a route may pass through.

    Two members joined only by a subject they share is a true connection and often the
    useful one; it is also how a small answer ends up lighting half the graph. The chip is
    the choice between the two, and it has to make a difference a reader can see.
    """
    page, errors = open_page(joined_only_through_a_topic())
    settle(page)
    assert page.get_attribute("#likewise", "aria-pressed") == "false"

    gather(page, "person:ada", "person:grace")
    assert page.locator("#picked button").count() == 2
    page.click("#findpath")
    page.wait_for_timeout(400)
    # through anything: the subject they share is what joins them, and it lights
    assert lit_kinds(page) == ["person", "person", "topic"]

    page.click("#likewise")
    page.wait_for_timeout(500)
    assert page.get_attribute("#likewise", "aria-pressed") == "true"
    # like with like: there is no way from one to the other through people alone
    assert "topic" not in lit_kinds(page)
    assert "unreachable" in page.text_content("#qcount")

    # and a pair that really is joined person-to-person still joins
    page.click("#qclear")
    gather(page, "person:grace")           # drop Grace, keep Ada
    gather(page, "person:alan")
    page.click("#findpath")
    page.wait_for_timeout(400)
    assert lit_kinds(page) == ["person", "person"]
    assert errors == []


def test_the_details_panel_can_go_back_wherever_you_came_from(open_page):
    """Fails when back only exists after an answer, or does not step back one at a time.

    Clicking a member, then one of their connections, then one of theirs, is how anyone
    reads a graph. Until this there was no way back but finding the first one again by eye,
    and the button only existed if an answer had put you there.
    """
    # a graph where a subject really does join two members, so there is a third hop to take
    page, errors = open_page(joined_only_through_a_topic())
    settle(page)
    # straight from the graph, with no answer anywhere: the first view has nothing behind it
    page.locator("#graph g.node", has_text="Ada Lovelace").first.press("Enter")
    page.wait_for_selector("#detail h2")
    assert page.text_content("#detail h2") == "Ada Lovelace"
    # The panel with nothing chosen is a view too — it is the only place the shared ground is
    # listed, so opening one of those pairs has to be undoable like any other step.
    home = page.locator("#detail #panel-back")
    assert home.count() == 1 and "What is here" in home.inner_text(), home.inner_text()
    home.click()
    page.wait_for_selector("#detail .placeholder")
    assert page.locator("#detail #panel-back").count() == 0, "home has nothing behind it"
    page.locator("#graph g.node", has_text="Ada Lovelace").first.press("Enter")
    page.wait_for_selector("#detail h2")

    # follow a connection, and back names where it came from
    page.click("#detail details.fold summary")
    page.locator('#detail .links button[data-id="topic:iron"]').click()
    page.wait_for_selector("#detail h2:text('iron')")
    back = page.locator("#detail #panel-back")
    assert back.count() == 1
    assert "Ada Lovelace" in back.inner_text()
    # it is at the top of the panel, above everything else in it
    box, panel = back.bounding_box(), page.locator("#detail").bounding_box()
    assert box["y"] - panel["y"] < 40 and box["x"] - panel["x"] < 40

    # one more hop, then back twice returns the way it came rather than jumping to the start
    page.click("#detail details.fold summary")
    page.locator('#detail .links button[data-id="person:grace"]').first.click()
    page.wait_for_selector("#detail h2:text('Grace Hopper')")
    page.click("#detail #panel-back")
    page.wait_for_selector("#detail h2:text('iron')")
    page.click("#detail #panel-back")
    page.wait_for_selector("#detail h2:text('Ada Lovelace')")
    assert "What is here" in page.locator("#detail #panel-back").inner_text()
    assert errors == []
