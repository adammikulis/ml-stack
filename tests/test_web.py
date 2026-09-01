"""The web as tools: search, read, look — with the network and the browser faked.

Every engine, fetch and browser here is a stand-in, so nothing reaches the network and no
browser opens. The one test that really searches is skipped unless ``MLSTACK_NET`` is set,
so a person can run it on purpose and an agent never does by accident.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import urllib.request

import pytest

from ml_stack import web
from ml_stack.scrape.browser import BrowserUnavailable
from ml_stack.web import (ENGINES, PROMPTS, SCHEMAS, Refused, SearchUnavailable, check, cut,
                          extract, look, read, search, searxng_engine, tools)

PAGE = ("<html><head><title>Quenlow Robotics - About</title></head><body><nav>Home</nav>"
        "<h1>About the Quenlow works</h1>"
        "<p>Quenlow Robotics builds arms for potteries. It was founded in Ambleford. "
        "The arms load kilns at night and unload them before anyone arrives.</p>"
        "<p>The company employs forty people, most of them in the Ambleford works, and "
        "sells to studios across the valley. Nothing it makes is sold to the public.</p>"
        "<script>var secret = 1;</script></body></html>")
SHELL = "<html><head><title>Quenlow</title></head><body><div id='app'></div></body></html>"
PNG = b"\x89PNG stub"


@pytest.fixture
def public_dns(monkeypatch):
    """Every invented host resolves somewhere public, except *.internal, which is a LAN."""
    def addresses(host):
        return ["10.0.0.5"] if host.endswith(".internal") else ["1.2.3.4"]
    monkeypatch.setattr(web, "_addresses", addresses)


def fetching(pages):
    """A fetcher that serves from a dict and remembers what was asked for."""
    asked = []

    def fetch(url):
        asked.append(url)
        if url not in pages:
            raise OSError(f"no such page {url}")
        return pages[url]
    fetch.asked = asked
    return fetch


class StubPage:
    """What web.py uses of a Playwright page, and a record of where it was sent."""

    def __init__(self, html, images=(), png=PNG):
        self.html, self.images, self.png = html, list(images), png
        self.visited = []

    def goto(self, url, **kw):
        self.visited.append(url)

    def content(self):
        return self.html

    def screenshot(self, **kw):
        return self.png

    def evaluate(self, script):
        return self.images


def browsing(page):
    """A ``browser()`` stand-in yielding one stub page."""
    @contextlib.contextmanager
    def browse():
        yield page
    return browse


def no_browser():
    raise BrowserUnavailable("playwright is not installed")


# --- search -------------------------------------------------------------------------------


def test_search_results_are_shaped_as_promised_and_capped():
    def engine(query, limit):
        assert query == "Quenlow Robotics" and limit == 2
        return [{"title": "  Quenlow  Robotics ", "url": "https://quenlow.example/",
                 "snippet": "arms for\npotteries", "extra": "dropped"},
                {"title": "dup", "url": "https://quenlow.example/", "snippet": ""},
                {"title": "no link", "url": "", "snippet": "skipped"},
                {"title": "second", "url": "https://pellard.example/", "snippet": "x"},
                {"title": "third", "url": "https://tessyn.example/", "snippet": "y"}]
    rows = search("  Quenlow   Robotics ", limit=2, engine=engine)
    assert rows == [{"title": "Quenlow Robotics", "url": "https://quenlow.example/",
                     "snippet": "arms for potteries"},
                    {"title": "second", "url": "https://pellard.example/", "snippet": "x"}]
    assert search("   ", engine=engine) == []


def test_the_engine_is_picked_by_env_and_an_unknown_name_is_refused(monkeypatch):
    seen = []
    monkeypatch.setitem(ENGINES, "pretend", lambda q, n: seen.append((q, n)) or [])
    monkeypatch.setenv("MLSTACK_SEARCH", "pretend")
    assert search("kilns", limit=3) == []
    assert seen == [("kilns", 3)]
    monkeypatch.setenv("MLSTACK_SEARCH", "nothing-called-this")
    with pytest.raises(SearchUnavailable, match="MLSTACK_SEARCH"):
        search("kilns")


def test_ddgs_rows_are_renamed_and_its_rate_limit_becomes_search_unavailable(monkeypatch):
    """ddgs 9 returns ``title/href/body`` and raises ``RatelimitException`` (a
    ``DDGSException``) rather than returning nothing; ``backend`` is still a keyword."""
    ddgs = pytest.importorskip("ddgs")
    from ddgs.exceptions import RatelimitException

    calls = []

    class FakeDDGS:
        def text(self, query, **kw):
            calls.append((query, kw))
            if query == "limited":
                raise RatelimitException("https://duckduckgo.example 202 Ratelimit")
            return [{"title": "About", "href": "https://quenlow.example/about",
                     "body": "arms for potteries"}]

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    monkeypatch.setenv("DDGS_BACKEND", "duckduckgo,brave")
    assert web.ddgs_engine("Quenlow", 4) == [
        {"title": "About", "url": "https://quenlow.example/about", "snippet": "arms for potteries"}]
    assert calls[-1] == ("Quenlow", {"max_results": 4, "backend": "duckduckgo,brave"})
    with pytest.raises(SearchUnavailable, match="RatelimitException"):
        web.ddgs_engine("limited", 4)


def test_searxng_builds_the_json_search_url_and_parses_the_reply(monkeypatch):
    opened = []

    def urlopen(request, timeout=None):
        opened.append((request.full_url, request.get_header("Accept")))
        body = json.dumps({"results": [
            {"title": "Pellard Foundry", "url": "https://pellard.example/",
             "content": "castings", "engine": "duckduckgo"},
            {"title": "second", "url": "https://tessyn.example/", "content": "c"}]})
        return contextlib.closing(io.BytesIO(body.encode()))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    # a self-hosted instance is on this side of the router, and must not be refused
    monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080/")
    rows = searxng_engine("Pellard Foundry", 1)
    assert rows == [{"title": "Pellard Foundry", "url": "https://pellard.example/",
                     "snippet": "castings"}]
    assert opened == [("http://localhost:8080/search?q=Pellard+Foundry&format=json",
                       "application/json")]

    monkeypatch.delenv("SEARXNG_URL")
    with pytest.raises(SearchUnavailable, match="SEARXNG_URL"):
        searxng_engine("anything", 1)


def test_a_searxng_that_is_down_is_search_unavailable_not_a_traceback(monkeypatch):
    def urlopen(request, timeout=None):
        raise OSError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("SEARXNG_URL", "http://searx.internal")
    with pytest.raises(SearchUnavailable, match="connection refused"):
        searxng_engine("kilns", 3)


# --- reading ------------------------------------------------------------------------------


def test_read_returns_title_and_text_and_never_the_scripts(public_dns):
    fetch = fetching({"https://quenlow.example/about": PAGE})
    got = read("https://quenlow.example/about", fetch=fetch, browse=no_browser)
    assert got["url"] == "https://quenlow.example/about"
    assert got["title"] == "About the Quenlow works"
    assert got["text"].startswith("Quenlow Robotics builds arms for potteries.")
    assert "secret" not in got["text"] and "Home" not in got["text"]
    assert got["rendered"] is False and "truncated" not in got


def test_read_cuts_on_a_sentence_boundary_and_says_so(public_dns):
    fetch = fetching({"https://quenlow.example/about": PAGE})
    got = read("https://quenlow.example/about", fetch=fetch, browse=no_browser, limit=150)
    assert got["truncated"] is True
    assert len(got["text"]) <= 150
    assert got["text"].endswith(".") and got["text"] == \
        "Quenlow Robotics builds arms for potteries. It was founded in Ambleford. " \
        "The arms load kilns at night and unload them before anyone arrives."


def test_cut_falls_back_to_a_space_and_leaves_short_text_alone():
    assert cut("short", 100) == ("short", False)
    words = " ".join(["word"] * 50)
    text, truncated = cut(words, 23)
    assert truncated and text == "word word word word" and not text.endswith(" ")
    # a sentence end too early in the window is not worth throwing the rest away for
    text, _ = cut("A. " + "b" * 40 + " " + "c" * 40, 60)
    assert text == "A. " + "b" * 40


def test_extract_without_trafilatura_strips_tags_itself(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "trafilatura", None)
    title, text = extract(PAGE)
    assert title == "Quenlow Robotics - About"
    assert "Quenlow Robotics builds arms for potteries." in text
    assert "secret" not in text
    assert "About the Quenlow works\n" in text, "block tags break lines"


@pytest.mark.parametrize("url", [
    "file:///etc/hosts",
    "ftp://quenlow.example/",
    "/just/a/path",
    "http://localhost/admin",
    "http://printer.local/",
    "http://127.0.0.1:8080/",
    "http://[::1]/",
    "http://10.1.2.3/",
    "http://192.168.1.1/",
    "http://172.16.0.9/",
    "http://169.254.169.254/latest/meta-data/",
    "http://intranet.internal/",
])
def test_a_url_on_this_side_of_the_router_is_refused_before_any_fetch(public_dns, url):
    fetch = fetching({})
    with pytest.raises(Refused):
        read(url, fetch=fetch, browse=no_browser)
    assert fetch.asked == []
    with pytest.raises(Refused):
        check(url)


def test_a_public_url_passes_the_check(public_dns):
    assert check("https://quenlow.example/about?x=1") == "https://quenlow.example/about?x=1"


def test_read_falls_through_to_the_browser_when_the_plain_text_is_thin(public_dns):
    """A script-built site serves a shell with a title and nothing else. That is under
    ``THIN`` characters, and it is the browser's page, not the shell, that gets read."""
    page = StubPage(PAGE)
    fetch = fetching({"https://quenlow.example/": SHELL})
    got = read("https://quenlow.example/", fetch=fetch, browse=browsing(page))
    assert page.visited == ["https://quenlow.example/"]
    assert got["rendered"] is True
    assert got["text"].startswith("Quenlow Robotics builds arms")


def test_read_does_not_open_a_browser_for_a_page_that_read_fine(public_dns):
    page = StubPage(SHELL)
    fetch = fetching({"https://quenlow.example/about": PAGE})
    got = read("https://quenlow.example/about", fetch=fetch, browse=browsing(page))
    assert page.visited == [] and got["rendered"] is False


def test_rendered_true_skips_the_plain_fetch(public_dns):
    page = StubPage(PAGE)
    fetch = fetching({})
    got = read("https://quenlow.example/", fetch=fetch, browse=browsing(page), rendered=True)
    assert fetch.asked == [] and page.visited == ["https://quenlow.example/"]
    assert got["rendered"] is True and got["title"] == "About the Quenlow works"


def test_without_a_browser_a_thin_page_is_returned_as_it_was(public_dns):
    fetch = fetching({"https://quenlow.example/": SHELL})
    got = read("https://quenlow.example/", fetch=fetch, browse=no_browser)
    assert got == {"url": "https://quenlow.example/", "title": "Quenlow", "text": "",
                   "rendered": False}


def test_a_failed_plain_fetch_is_raised_when_the_browser_cannot_help(public_dns):
    with pytest.raises(OSError, match="no such page"):
        read("https://quenlow.example/gone", fetch=fetching({}), browse=no_browser)


# --- looking ------------------------------------------------------------------------------


def test_look_returns_the_screenshot_first_then_the_largest_pictures(public_dns):
    page = StubPage(PAGE, images=[
        {"src": "https://quenlow.example/img/small.png", "width": 199, "height": 800},
        {"src": "data:image/png;base64,AAAA", "width": 900, "height": 900},
        {"src": "/img/medium.jpg", "width": 400, "height": 300},
        {"src": "https://quenlow.example/img/big.jpg", "width": 1200, "height": 800},
        {"src": "https://quenlow.example/img/big.jpg", "width": 1200, "height": 800},
        {"src": "https://quenlow.example/img/third.jpg", "width": 300, "height": 300},
        {"src": "https://quenlow.example/img/fourth.jpg", "width": 250, "height": 250},
        {"src": "https://quenlow.example/img/fifth.jpg", "width": 210, "height": 210},
    ])
    fetched = []

    def fetch_bytes(url):
        fetched.append(url)
        return url.rsplit("/", 1)[1].encode()

    got = look("https://quenlow.example/about", browse=browsing(page), fetch_bytes=fetch_bytes)
    assert set(got) == {"url", "title", "text", "_images"}
    assert got["title"] == "About the Quenlow works"
    assert got["text"].startswith("Quenlow Robotics builds arms") and len(got["text"]) <= 1500
    assert got["_images"] == [PNG, b"big.jpg", b"medium.jpg", b"third.jpg"]
    assert fetched == ["https://quenlow.example/img/big.jpg",
                       "https://quenlow.example/img/medium.jpg",
                       "https://quenlow.example/img/third.jpg"]


def test_look_keeps_going_when_a_picture_will_not_come(public_dns):
    page = StubPage(PAGE, images=[
        {"src": "https://quenlow.example/img/gone.jpg", "width": 900, "height": 900},
        {"src": "https://quenlow.example/img/here.jpg", "width": 300, "height": 300}])

    def fetch_bytes(url):
        if "gone" in url:
            raise OSError("404")
        return b"here"

    got = look("https://quenlow.example/", browse=browsing(page), fetch_bytes=fetch_bytes)
    assert got["_images"] == [PNG, b"here"]


def test_look_refuses_a_private_url_before_the_browser_goes_anywhere(public_dns):
    page = StubPage(PAGE)
    with pytest.raises(Refused):
        look("http://127.0.0.1/", browse=browsing(page), fetch_bytes=lambda u: b"")
    with pytest.raises(Refused):
        look("http://intranet.internal/", browse=browsing(page), fetch_bytes=lambda u: b"")
    assert page.visited == []


def test_a_picture_on_a_private_host_is_skipped_not_fetched(public_dns):
    page = StubPage(PAGE, images=[
        {"src": "http://intranet.internal/badge.png", "width": 900, "height": 900}])
    got = look("https://quenlow.example/", browse=browsing(page))   # the real fetcher
    assert got["_images"] == [PNG]


# --- the tools ----------------------------------------------------------------------------


def names(pairs):
    return [schema["function"]["name"] for schema, _ in pairs]


def test_two_tools_for_a_text_model_and_three_for_one_that_sees():
    assert names(tools()) == ["web_search", "web_read"]
    assert names(tools(vision=True)) == ["web_search", "web_read", "web_look"]
    for schema, callable_ in tools(vision=True):
        assert callable(callable_) and schema["type"] == "function"
        assert schema["function"]["parameters"]["required"]


def test_every_description_carries_worked_examples_and_says_when_not_to():
    for schema in SCHEMAS:
        said = schema["function"]["description"]
        assert said.count("→") >= 2, schema["function"]["name"]
        assert f"{schema['function']['name']}(" in said, "the example must be a call"
        assert "not" in said.casefold(), "each says when not to use it"
    assert "look_up" in SCHEMAS[0]["function"]["description"], \
        "web_search says the graph comes first"
    assert "web_read" in SCHEMAS[2]["function"]["description"], \
        "web_look says to use web_read for words"


def test_prompts_are_questions_for_every_tool_and_none_reach_the_model():
    assert set(PROMPTS) == set(names(tools(vision=True)))
    assert len(PROMPTS["web_search"]) >= 5 and len(PROMPTS["web_read"]) >= 3
    said = " ".join(s["function"]["description"] for s in SCHEMAS)
    for examples in PROMPTS.values():
        for example in examples:
            assert example not in said


def test_the_search_tool_says_so_when_the_engine_will_not_answer():
    def limited(query, limit):
        raise SearchUnavailable("202 Ratelimit")
    pairs = dict(zip(names(tools(engine=limited)), (c for _, c in tools(engine=limited))))
    got = pairs["web_search"]({"query": "Quenlow Robotics"})
    assert got == {"none": "search unavailable: 202 Ratelimit"}
    assert pairs["web_search"]({"query": ""}) == {"none": "nothing to search for: pass a query"}


def test_the_search_tool_says_nothing_matched_rather_than_returning_a_list():
    pairs = dict(zip(names(tools(engine=lambda q, n: [])), (c for _, c in tools(engine=lambda q, n: []))))
    got = pairs["web_search"]({"query": "kilns"})
    assert "none" in got and "kilns" in got["none"]
    pairs = dict(zip(names(tools(engine=lambda q, n: [{"title": "t", "url": "https://tessyn.example/", "snippet": "s"}])),
                     (c for _, c in tools(engine=lambda q, n: [{"title": "t", "url": "https://tessyn.example/", "snippet": "s"}]))))
    assert pairs["web_search"]({"query": "kilns"}) == [
        {"title": "t", "url": "https://tessyn.example/", "snippet": "s"}]


def test_the_read_tool_turns_a_refusal_and_a_failure_into_none(public_dns):
    fetch = fetching({"https://quenlow.example/about": PAGE})
    (_, _), (_, reading) = tools(fetch=fetch, browse=no_browser)
    assert reading({"url": "http://127.0.0.1/"})["none"].startswith("could not read")
    assert "none" in reading({"url": "https://quenlow.example/missing"})
    got = reading({"url": "https://quenlow.example/about"})
    assert got["title"] == "About the Quenlow works" and got["rendered"] is False


def test_the_read_tool_passes_rendered_through(public_dns):
    page = StubPage(PAGE)
    (_, _), (_, reading) = tools(fetch=fetching({}), browse=browsing(page))
    got = reading({"url": "https://quenlow.example/", "rendered": True})
    assert got["rendered"] is True and page.visited == ["https://quenlow.example/"]


def test_the_look_tool_says_no_browser_without_one(public_dns):
    (_, _), (_, _), (_, looking) = tools(browse=no_browser, vision=True)
    got = looking({"url": "https://quenlow.example/"})
    assert got == {"none": "no browser: playwright is not installed"}
    assert looking({"url": "http://localhost/"})["none"].startswith("could not look at")


def test_the_look_tool_returns_the_shape_the_ask_loop_strips(public_dns):
    page = StubPage(PAGE, images=[])
    (_, _), (_, _), (_, looking) = tools(browse=browsing(page), vision=True)
    got = looking({"url": "https://quenlow.example/"})
    assert got["_images"] == [PNG] and got["title"] == "About the Quenlow works"


@pytest.mark.skipif(not os.environ.get("MLSTACK_NET"),
                    reason="really searches the web; set MLSTACK_NET=1 to run on purpose")
def test_a_real_search_returns_pages_with_links():
    """Deliberate: this one reaches ddgs and whatever engines it fronts. A rate limit is a
    skip, not a failure — it says something about the afternoon, not about the code."""
    try:
        rows = search("pottery kiln firing schedule", limit=3)
    except SearchUnavailable as exc:
        pytest.skip(f"the engine would not answer: {exc}")
    assert rows and all(r["url"].startswith("http") for r in rows)
    assert set(rows[0]) == {"title", "url", "snippet"}
