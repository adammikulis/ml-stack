"""Following a listing across pages, under a budget, against a real local server."""

from __future__ import annotations

import re
import urllib.parse

import pytest

from ml_stack.scrape.crawl import Walk, crawl, links, more, next_link, pages
from ml_stack.scrape.polite import Disallowed, Polite
from tests.web_site import allow_all, serving


@pytest.fixture
def site():
    yield from serving()


def quick(**kw) -> Polite:
    return Polite(**{"guard": allow_all, "min_interval_s": 0.0, "backoff_s": 0.0,
                     "robots": False, **kw})


def listing(n: int, *, total: int, nav: str) -> str:
    rows = "".join(f'<li><a class="item" href="/part/{i}">Part {i}</a></li>'
                   for i in range((n - 1) * 3, min(n * 3, total)))
    return f"<html><body><ul>{rows}</ul>{nav}</body></html>"


def parts(html: str, url: str):
    base = urllib.parse.urlsplit(url)
    return [{"url": f"{base.scheme}://{base.netloc}{m}", "title": t}
            for m, t in re.findall(r'class="item" href="([^"]+)">([^<]+)<', html)]


def test_rel_next_wins_over_everything_else():
    html = ('<link rel="next" href="/a?page=2"><a href="/x">Next</a>')
    assert next_link(html, "http://h/a") == "http://h/a?page=2"
    assert next_link('<a rel="next nofollow" href="/b">go</a>', "http://h/a") == "http://h/b"


@pytest.mark.parametrize("text", ["Next", "next page", "\u203a", "\u00bb", "Older posts"])
def test_a_next_looking_anchor_is_followed(text):
    assert next_link(f'<a href="/p/2">{text}</a><a href="/p/0">Previous</a>', "http://h/p/1") \
        == "http://h/p/2"


def test_an_aria_labelled_arrow_is_followed():
    assert next_link('<a href="/p/2" aria-label="Next page"><svg></svg></a>', "http://h/p/1") \
        == "http://h/p/2"


def test_numbered_links_pick_the_smallest_page_above_this_one():
    html = '<a href="/l?page=1">1</a><a href="/l?page=3">3</a><a href="/l?page=2">2</a>'
    assert next_link(html, "http://h/l?page=1") == "http://h/l?page=2"
    assert next_link(html, "http://h/l") == "http://h/l?page=2"


def test_offset_links_step_by_whatever_the_page_uses():
    html = '<a href="/l?offset=0">1</a><a href="/l?offset=20">2</a><a href="/l?offset=40">3</a>'
    assert next_link(html, "http://h/l?offset=20") == "http://h/l?offset=40"


def test_a_page_param_with_no_anchors_is_guessed_one_ahead():
    assert next_link("<p>no links</p>", "http://h/l?q=a&page=4") == "http://h/l?q=a&page=5"
    assert next_link("<p>no links</p>", "http://h/l?page=4", guess=False) == ""
    assert next_link("<p>no links</p>", "http://h/l") == ""


def test_the_last_numbered_page_has_no_next():
    html = '<a href="/l?page=1">1</a><a href="/l?page=2">2</a>'
    assert next_link(html, "http://h/l?page=2") == ""


def test_a_guessed_page_that_is_not_there_ends_the_walk(site):
    site.page("/g?page=1", "<p>one</p>")
    got = list(pages(f"{site.base}/g?page=1", polite=quick(), walk=Walk(max_pages=5)))
    assert len(got) == 1 and site.hits == ["/g?page=1", "/g?page=2"]


def test_a_numbered_link_to_a_different_listing_is_not_the_next_page():
    html = '<a href="/other?page=2">2</a>'
    assert next_link(html, "http://h/l?page=1", guess=False) == ""


def test_links_reports_text_and_hint():
    found, hinted = links('<link rel=next href="/n"><a href="/a" title="T"> A  b </a>', "http://h/")
    assert hinted == "http://h/n"
    assert found[0].text == "A b" and found[0].label == "T"


def test_a_listing_is_followed_to_its_end(site):
    for n in (1, 2, 3):
        nav = f'<a rel="next" href="/list?page={n + 1}">Next</a>' if n < 3 else ""
        site.page(f"/list?page={n}", listing(n, total=8, nav=nav))
    site.routes["/list"] = site.routes["/list?page=1"]
    got = crawl(f"{site.base}/list", items=parts, polite=quick(), walk=Walk(max_pages=10))
    assert [r["title"] for r in got] == [f"Part {i}" for i in range(8)]
    assert site.hits == ["/list", "/list?page=2", "/list?page=3", "/list?page=4"]


def test_the_page_budget_stops_the_walk(site):
    for n in range(1, 6):
        site.page(f"/list?page={n}", listing(n, total=15, nav=f'<a rel="next" href="/list?page={n + 1}">Next</a>'))
    got = list(pages(f"{site.base}/list?page=1", items=parts, polite=quick(), walk=Walk(max_pages=2)))
    assert [p.number for p in got] == [1, 2]
    assert len(site.hits) == 2


def test_the_item_budget_cuts_mid_page(site):
    for n in range(1, 4):
        site.page(f"/list?page={n}", listing(n, total=9, nav=f'<a rel="next" href="/list?page={n + 1}">Next</a>'))
    got = crawl(f"{site.base}/list?page=1", items=parts, polite=quick(), walk=Walk(max_items=4))
    assert len(got) == 4
    assert len(site.hits) == 2


def test_a_page_of_only_repeats_ends_the_walk(site):
    same = listing(1, total=3, nav='<a rel="next" href="/list?page=2">Next</a>')
    site.page("/list?page=1", same)
    site.page("/list?page=2", same.replace("page=2", "page=3"))
    site.page("/list?page=3", "never")
    got = crawl(f"{site.base}/list?page=1", items=parts, polite=quick(), walk=Walk(max_pages=9))
    assert len(got) == 3
    assert "/list?page=3" not in site.hits


def test_a_loop_of_next_links_visits_each_url_once(site):
    site.page("/a", '<a rel="next" href="/b">n</a><p>a</p>')
    site.page("/b", '<a rel="next" href="/a">n</a><p>b</p>')
    got = list(pages(f"{site.base}/a", polite=quick(), walk=Walk(max_pages=9)))
    assert [p.url.rsplit("/", 1)[1] for p in got] == ["a", "b"]


def test_a_seen_set_carries_over_between_walks(site):
    site.page("/list", listing(1, total=3, nav=""))
    held: set[str] = set()
    first = crawl(f"{site.base}/list", items=parts, polite=quick(), walk=Walk(seen=held))
    again = crawl(f"{site.base}/list", items=parts, polite=quick(), walk=Walk(seen=held))
    assert len(first) == 3 and again == []


def test_requests_to_one_host_are_spaced(site):
    for n in (1, 2, 3):
        site.page(f"/s{n}", f'<a rel="next" href="/s{n + 1}">n</a>')
    list(pages(f"{site.base}/s1", polite=quick(min_interval_s=0.15), walk=Walk(max_pages=3)))
    gaps = [b - a for a, b in zip(site.times, site.times[1:], strict=False)]
    assert len(gaps) == 2 and min(gaps) >= 0.14


def test_robots_txt_that_disallows_stops_the_fetch(site):
    site.file("/robots.txt", b"User-agent: *\nDisallow: /private\n", "text/plain")
    site.page("/private/x", "secret")
    site.page("/public", "ok")
    polite = quick(robots=True)
    assert polite.allowed(f"{site.base}/public")
    with pytest.raises(Disallowed):
        list(pages(f"{site.base}/private/x", polite=polite))
    assert "/private/x" not in site.hits
    assert polite.note(f"{site.base}/private/x") == "robots.txt disallows"


def test_robots_off_does_not_look(site):
    site.page("/p", "ok")
    list(pages(f"{site.base}/p", polite=quick(robots=False)))
    assert "/robots.txt" not in site.hits


def test_a_missing_robots_txt_allows_and_an_erroring_one_forbids(site):
    assert quick(robots=True).allowed(f"{site.base}/anything")
    site.routes["/robots.txt"] = (503, {}, b"down")
    assert not quick(robots=True, tries=1).allowed(f"{site.base}/anything")


def test_a_503_is_retried_then_succeeds(site):
    calls = []

    def flaky(path):
        calls.append(path)
        return (503, {}, b"busy") if len(calls) < 3 else (200, {"Content-Type": "text/html"}, b"<p>ok</p>")
    site.routes["/flaky"] = flaky
    got = list(pages(f"{site.base}/flaky", polite=quick(tries=4)))
    assert got[0].html == "<p>ok</p>" and len(calls) == 3


def test_crawl_refuses_private_hosts_by_default(site):
    from ml_stack.http import Refused
    with pytest.raises(Refused):
        list(pages(f"{site.base}/x", fetch=None, polite=Polite(robots=False)))


class Button:
    def __init__(self, present):
        self.present = present

    def count(self):
        return 1 if self.present() else 0

    @property
    def first(self):
        return self

    def click(self):
        self.on_click()


class LoadMorePage:
    """A listing that grows by three rows per click, three clicks in all."""

    url = "http://h/list"

    def __init__(self):
        self.shown = 3
        self.clicks = 0
        self.button = Button(lambda: self.shown < 9)
        self.button.on_click = self.click

    def click(self):
        self.shown += 3
        self.clicks += 1

    def content(self):
        return "<ul>" + "".join(f'<li><a class="item" href="/part/{i}">Part {i}</a></li>'
                                for i in range(self.shown)) + "</ul>"

    def get_by_role(self, role, name=None):
        return self.button

    def get_by_text(self, name):
        return Button(lambda: False)

    def wait_for_timeout(self, ms):
        pass


def test_load_more_clicks_until_the_button_is_gone():
    page = LoadMorePage()
    got = list(more(page, items=parts, settle_ms=0))
    assert [len(p.items) for p in got] == [3, 3, 3]
    assert page.clicks == 2


def test_load_more_stops_at_its_click_budget():
    page = LoadMorePage()
    got = list(more(page, items=parts, max_clicks=1, settle_ms=0))
    assert page.clicks == 1 and len(got) == 2
