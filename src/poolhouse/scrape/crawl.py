"""Following a listing across its pages: the 'next' link, a page or offset number, a 'load
more' button, each under a budget."""

from __future__ import annotations

import contextlib
import json
import re
import urllib.parse
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from poolhouse.http import ServerError
from poolhouse.scrape.polite import Polite

PAGE_PARAMS = ("page", "pagenum", "pageno", "p", "pg", "paged")
OFFSET_PARAMS = ("offset", "start", "from", "skip", "first")
NEXT_TEXT = re.compile(r"^\s*(next(\s+page)?|older(\s+posts)?|[>\u203a\u00bb\u2192]+|next\s*[>\u203a\u00bb\u2192]+)\s*$",
                       re.IGNORECASE)
MORE_TEXT = re.compile(r"(load|show|view|see)\s+more|more\s+(results|items)", re.IGNORECASE)
MOST_BYTES = 8 * 1024 * 1024

Items = Callable[[str, str], list[dict[str, Any]]]
"""``(html, url) -> rows``: what one page of a listing holds."""


@dataclass(frozen=True)
class Link:
    """One anchor on a page: where it goes, what it says, and how it is marked up."""

    href: str
    text: str
    rel: str = ""
    label: str = ""


class _Anchors(HTMLParser):
    def __init__(self, base: str) -> None:
        super().__init__()
        self.base = base
        self.found: list[Link] = []
        self.hinted = ""
        self._open: dict[str, str] | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        a = {k: v or "" for k, v in attrs}
        if tag == "link" and "next" in a.get("rel", "").lower().split() and a.get("href"):
            self.hinted = urllib.parse.urljoin(self.base, a["href"])
        elif tag == "a" and a.get("href"):
            self._open, self._text = a, []

    def handle_data(self, data: str) -> None:
        if self._open is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._open is not None:
            a = self._open
            self.found.append(Link(
                href=urllib.parse.urljoin(self.base, a["href"]),
                text=" ".join("".join(self._text).split()),
                rel=a.get("rel", "").lower(),
                label=a.get("aria-label", "") or a.get("title", "")))
            self._open = None


def links(html: str, url: str) -> tuple[list[Link], str]:
    """Every anchor on the page, and the ``<link rel="next">`` target ("" when absent)."""
    parser = _Anchors(url)
    with contextlib.suppress(Exception):
        parser.feed(html)
        parser.close()
    return parser.found, parser.hinted


def _with(url: str, name: str, value: int) -> str:
    parts = urllib.parse.urlsplit(url)
    query = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
             if k != name]
    query.append((name, str(value)))
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def _param(url: str, names: tuple[str, ...]) -> tuple[str, int] | None:
    for key, value in urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query):
        if key.lower() in names and value.isdigit():
            return key, int(value)
    return None


def _same_listing(a: str, b: str, name: str) -> bool:
    """Whether two URLs differ at most in the query parameter ``name``."""
    pa, pb = urllib.parse.urlsplit(a), urllib.parse.urlsplit(b)
    if (pa.netloc, pa.path) != (pb.netloc, pb.path):
        return False

    def rest(p: urllib.parse.SplitResult) -> list[tuple[str, str]]:
        return sorted((k, v) for k, v in urllib.parse.parse_qsl(p.query) if k != name)
    return rest(pa) == rest(pb)


def next_link(html: str, url: str, *, guess: bool = True) -> str:
    """The address of the page after this one, or "" when there is none.

    In order: ``<link rel=next>`` or ``<a rel=next>``; an anchor reading Next, Older,
    or an arrow (or labelled so); a numbered anchor on the same listing whose ``page=``
    or ``offset=`` is the smallest above this page's; and, with ``guess``, this URL's own
    page number plus one when the URL carries one and the page links to no other page number.
    """
    found, hinted = links(html, url)
    if hinted:
        return hinted
    for link in found:
        if "next" in link.rel.split():
            return link.href
    for link in found:
        if NEXT_TEXT.match(link.text) or NEXT_TEXT.match(link.label) \
                or re.match(r"^next\b", link.label, re.IGNORECASE):
            return link.href
    for names, base in ((PAGE_PARAMS, 1), (OFFSET_PARAMS, 0)):
        current = _param(url, names)
        name = current[0] if current else None
        number = current[1] if current else base
        best: tuple[int, str] | None = None
        for link in found:
            seen = _param(link.href, names)
            if seen is None or (name and seen[0] != name):
                continue
            if seen[1] > number and _same_listing(url, link.href, seen[0]) \
                    and (best is None or seen[1] < best[0]):
                best = (seen[1], link.href)
        if best:
            return best[1]
    paged = _param(url, PAGE_PARAMS)
    if guess and paged and not any(_param(link.href, PAGE_PARAMS) for link in found):
        return _with(url, paged[0], paged[1] + 1)
    return ""


@dataclass
class Page:
    """One page of a listing: where it was, what it said, and the rows new to this walk."""

    number: int
    url: str
    html: str
    items: list[dict[str, Any]] = field(default_factory=list)
    next: str = ""


def _key(row: dict[str, Any]) -> str:
    if row.get("url"):
        return str(row["url"])
    return json.dumps(row, sort_keys=True, default=str)


@dataclass
class Walk:
    """How far a walk goes and what it has already seen.

    ``max_pages`` and ``max_items`` bound it; rows whose ``url`` (or whole content, when
    they have none) is in ``seen`` are dropped, and ``seen`` is added to, so a caller that
    keeps the set skips them next time. ``follow(html, url)`` finds the next address.
    """

    max_pages: int = 5
    max_items: int | None = None
    seen: set[str] = field(default_factory=set)
    follow: Callable[[str, str], str] = next_link


def pages(start: str, items: Items | None = None, walk: Walk | None = None,
          polite: Polite | None = None, fetch: Callable[[str], str] | None = None
          ) -> Iterator[Page]:
    """Walk a listing from ``start``, yielding each page, until a budget or the end.

    ``items`` pulls rows from a page's HTML; ``walk`` sets the budget and holds what was
    seen, and a page that gives no new row ends the walk. ``polite`` spaces the requests and
    honours robots.txt (a default ``Polite()`` when None); ``fetch(url) -> html`` replaces
    the network. A URL is never fetched twice, and a 404 after the first page ends the walk.
    """
    polite = polite or Polite()
    walk = walk or Walk()
    visited: set[str] = set()
    url, taken = start, 0
    for number in range(1, walk.max_pages + 1):
        if not url or url in visited:
            return
        visited.add(url)
        try:
            html = fetch(url) if fetch else _body(polite, url)
        except ServerError as exc:
            if number > 1 and exc.status in (404, 410):
                return
            raise
        rows: list[dict[str, Any]] = []
        for row in (items(html, url) if items else []):
            marker = _key(row)
            if marker in walk.seen:
                continue
            walk.seen.add(marker)
            rows.append(row)
        if walk.max_items is not None:
            rows = rows[:max(0, walk.max_items - taken)]
        taken += len(rows)
        ahead = walk.follow(html, url)
        yield Page(number=number, url=url, html=html, items=rows, next=ahead)
        if items and not rows:
            return
        if walk.max_items is not None and taken >= walk.max_items:
            return
        url = ahead


def crawl(start: str, items: Items, walk: Walk | None = None, polite: Polite | None = None
          ) -> list[dict[str, Any]]:
    """Every row ``pages`` finds from ``start``, in order."""
    return [row for page in pages(start, items, walk, polite) for row in page.items]


def _body(polite: Polite, url: str) -> str:
    with polite.fetch(url, accept="text/html,application/xhtml+xml,*/*;q=0.5") as reply:
        raw = reply.read(MOST_BYTES)
        charset = reply.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, "replace")


def more(page: Any, *, items: Items, max_clicks: int = 10, settle_ms: int = 800,
         button: re.Pattern[str] = MORE_TEXT) -> Iterator[Page]:
    """Click a 'load more' button until it is gone or the page stops growing.

    ``page`` is a Playwright page already on the listing. Yields the page's HTML after each
    load, with only the rows not yielded before in ``items``.
    """
    seen: set[str] = set()
    last = -1
    for number in range(1, max_clicks + 2):
        html = str(page.content())
        rows = []
        for row in items(html, str(page.url)):
            marker = _key(row)
            if marker not in seen:
                seen.add(marker)
                rows.append(row)
        yield Page(number=number, url=str(page.url), html=html, items=rows)
        if len(html) == last or number > max_clicks:
            return
        last = len(html)
        target = page.get_by_role("button", name=button)
        if target.count() == 0:
            target = page.get_by_text(button)
        if target.count() == 0:
            return
        target.first.click()
        page.wait_for_timeout(settle_ms)
