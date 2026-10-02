"""The web, as tools a model can call beside the graph's.

The four tools in ``graph.looking`` see the graph and nothing else, which is right for "who
here does robotics" and useless for "what does that company actually do" or "is there a
newer release of this". ``tools()`` here gives a model two more pairs in the same
``(schema, callable)`` shape — ``web_search`` and ``web_read`` — and a third, ``web_look``,
for a model that can see: a screenshot and the page's biggest pictures.

What is deliberate about the shape:

- **Search and reading are separate calls**, so a model reads one page it chose rather
  than eight it did not. Reading is the expensive half, in seconds and in context.
- **A search that will not answer says so** — ``SearchUnavailable`` from the function,
  ``{"none": "..."}`` from the tool — rather than returning ``[]``, because an empty list
  reads to a model as "try again" and a reason reads as "move on". That mirrors ``find``
  in ``graph.looking``, and the measurement behind it is there.
- **A model must not be able to read the machine it runs on.** ``read`` refuses anything
  that is not http(s), and any host that resolves to a loopback, private or link-local
  address, before a byte is fetched or a browser navigates. The check is on what the name
  *resolves to*, not what it looks like, because ``localhost`` is spelt many ways.
- **The examples in the descriptions are the point.** Measured in ``graph.prompts``: a worked
  call in the description took a 4B model from 17% to 70% recall on the same weights,
  where prompt text had not. Every example below is invented, and the sites are under
  the reserved ``.example`` domain so none of them can be fetched by accident.

Search comes from ``ddgs`` by default (keyless; a metasearch over several engines, and
rate-limited by them) or a self-hosted SearXNG, chosen by ``MLSTACK_SEARCH``. Reading uses
``trafilatura`` when installed and a stdlib tag-stripper when not. Rendering and
screenshots use ``ml_stack.scrape.browser``, which needs playwright; without it ``read``
returns the plain text and ``web_look`` says there is no browser.
"""

from __future__ import annotations

import contextlib
import inspect
import itertools
import json
import os
import urllib.parse
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from ml_stack import net
from ml_stack.home import cache, state
from ml_stack.http import Refused, ServerError, check
from ml_stack.markup import cut, extract
from ml_stack.net import pdftext, untrusted
from ml_stack.scrape.crawl import next_link
from ml_stack.scrape.download import KINDS, LICENCE, Download, Wanted, as_dict, download
from ml_stack.scrape.polite import Polite

Engine = Callable[..., list[dict[str, Any]]]
"""``(query, limit) -> [{"title", "url", "snippet"}, ...]``; may raise SearchUnavailable.
An engine that can page also takes ``page`` (1-based) as a third argument."""

INSTALL = "pip install 'ml-stack[web]'"
# what a plain fetch has to come back with before it counts as having read the page:
# a script-built site serves a shell with a title and nothing else, and that shell is
# what falls through to the browser
THIN = 200
LOOK_CHARS = 1500
LEAST_IMAGE = 200
MOST_IMAGES = 3
MOST_BYTES = 8 * 1024 * 1024
TIMEOUT_S = 20.0
USER_AGENT = "Mozilla/5.0 (compatible; ml-stack)"
# its own profile, not the scraper's: the scraper's profile is signed in to the community's
# workspace, and a page a model chose to open must never carry those cookies
def profile_dir() -> Path:
    """Where the window this library opens keeps its own browser profile."""
    return state("web")


class SearchUnavailable(RuntimeError):
    """The search engine would not answer: rate-limited, timed out, offline, or missing."""


def downloads_dir() -> Path:
    """Where ``fetch_file`` and PDF reads keep what they download."""
    return cache("web", "downloads")


def politeness() -> Polite:
    """Request manners for what this module fetches itself: ``MLSTACK_ROBOTS=off`` stops
    consulting robots.txt, ``MLSTACK_MIN_INTERVAL`` sets the seconds between requests to a host."""
    gap = os.environ.get("MLSTACK_MIN_INTERVAL") or "1.0"
    return Polite(robots=(os.environ.get("MLSTACK_ROBOTS") or "on").lower() != "off",
                  min_interval_s=float(gap))


# --- search -------------------------------------------------------------------------------


def ddgs_engine(query: str, limit: int, page: int = 1) -> list[dict[str, Any]]:
    """Search through ``ddgs`` — keyless, several engines behind one call.

    ``DDGS_BACKEND`` picks the engines (``auto`` by default, or a comma list such as
    ``duckduckgo,brave``). ddgs raises its own exceptions for a rate limit, a timeout and
    for "no results" alike; all of them become ``SearchUnavailable`` here, with the reason.
    """
    try:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException
    except ImportError as exc:
        raise ImportError(f"ddgs is not installed: {INSTALL}") from exc
    backend = os.environ.get("DDGS_BACKEND") or "auto"
    try:
        rows = DDGS().text(query, max_results=limit, backend=backend,
                             **({"page": page} if page > 1 else {}))
    except DDGSException as exc:
        raise SearchUnavailable(f"{type(exc).__name__}: {exc}") from exc
    return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in rows]


def searxng_engine(query: str, limit: int, page: int = 1) -> list[dict[str, Any]]:
    """Search a SearXNG instance at ``SEARXNG_URL`` through its ``/search?format=json``.

    Self-hosted, so nobody rate-limits it but its owner; the JSON format has to be enabled
    in the instance's ``settings.yml``. Stdlib only.
    """
    base = (os.environ.get("SEARXNG_URL") or "").rstrip("/")
    if not base:
        raise SearchUnavailable("SEARXNG_URL is not set")
    url = f"{base}/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", **({"pageno": page} if page > 1 else {})})
    try:
        host = urllib.parse.urlsplit(url).hostname or ""
        body = _http(url, accept="application/json", private=(host,))
        payload = json.loads(body.decode("utf-8", "replace"))
    except (ServerError, OSError, ValueError) as exc:
        raise SearchUnavailable(f"searxng at {base}: {exc}") from exc
    rows = payload.get("results") if isinstance(payload, Mapping) else None
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in (rows or [])[:limit] if isinstance(r, Mapping)]


ENGINES: dict[str, Engine] = {"ddgs": ddgs_engine, "searxng": searxng_engine}


def search(query: str, *, limit: int = 8, engine: Engine | None = None, page: int = 1
           ) -> list[dict[str, Any]]:
    """Pages about ``query``: ``[{"title", "url", "snippet"}, ...]``, at most ``limit``.

    ``engine`` is ``(query, limit) -> rows``; when None, ``ENGINES[MLSTACK_SEARCH]`` with
    ``ddgs`` the default. ``page`` (1-based) asks for a later page of results, which needs an
    engine that takes it. A blank query finds nothing. Raises ``SearchUnavailable`` when the
    engine would not answer, and ``ImportError`` when it is not installed.
    """
    wanted = " ".join((query or "").split())
    if not wanted:
        return []
    if engine is None:
        name = os.environ.get("MLSTACK_SEARCH") or "ddgs"
        try:
            engine = ENGINES[name]
        except KeyError:
            raise SearchUnavailable(
                f"MLSTACK_SEARCH={name!r} is not one of {', '.join(sorted(ENGINES))}") from None
    if page > 1 and "page" not in inspect.signature(engine).parameters:
        raise SearchUnavailable("this search engine does not take a page number")
    rows = engine(wanted, limit, page) if page > 1 else engine(wanted, limit)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        url = str(row.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({"title": " ".join(untrusted.clean_text(str(row.get("title") or ""))[0].split()),
                    "url": url,
                    "snippet": " ".join(untrusted.clean_text(str(row.get("snippet") or ""))[0].split())})
        if len(out) >= limit:
            break
    return out


def search_pages(query: str, *, pages: int = 3, limit: int = 8, engine: Engine | None = None
                 ) -> Iterator[list[dict[str, Any]]]:
    """Results page by page, each without the URLs an earlier page gave, until ``pages`` or a
    page with nothing new. A later page the engine will not give ends the stream quietly."""
    seen: set[str] = set()
    for number in range(1, pages + 1):
        try:
            rows = search(query, limit=limit, engine=engine, page=number)
        except SearchUnavailable:
            if number == 1:
                raise
            return
        fresh = [r for r in rows if r["url"] not in seen]
        seen.update(r["url"] for r in fresh)
        if not fresh:
            return
        yield fresh


# --- fetching -------------------------------------------------------------------------------


def _http(url: str, *, accept: str = "*/*", most: int = MOST_BYTES, private: tuple[str, ...] = (),
          admit: bool = False) -> bytes:
    """One GET through the net pipeline with a size cap; ``private`` names a host on this
    network that a search backend lives on."""
    ask = net.Ask(purpose="web", admit=admit, tries=3, max_bytes=most, private=private,
                  headers={"User-Agent": USER_AGENT, "Accept": accept})
    with net.default().open(url, ask) as reply:
        return reply.read(most)


def _get(url: str, *, accept: str = "*/*", most: int = MOST_BYTES) -> bytes:
    """One GET of a page a model chose, with a size cap."""
    return _http(url, accept=accept, most=most)


def _fetch(url: str) -> str:
    """A page's HTML."""
    body = _get(url, accept="text/html,application/xhtml+xml,*/*;q=0.5")
    return body.decode("utf-8", "replace")


def _fetch_bytes(url: str) -> bytes:
    """A picture's bytes, through the same pipeline as a page."""
    return _get(url, accept="image/*")


# --- reading ------------------------------------------------------------------------------


def _browse() -> Any:
    """A page in a real browser, as a context manager; ``BrowserUnavailable`` without playwright."""
    from ml_stack.scrape.browser import Window, browser

    return browser(Window(profile=profile_dir(), guarded=True))


# Removes every element a reader could not see: not displayed, hidden, transparent, smaller than
# a point, off the page, or the same colour as what is behind it.
_UNSEEN_JS = """() => {
  const back = (el) => { for (let n = el; n; n = n.parentElement) {
    const c = getComputedStyle(n).backgroundColor;
    if (c && c !== 'rgba(0, 0, 0, 0)' && c !== 'transparent') return c; } return 'rgb(255, 255, 255)'; };
  const gone = [];
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el), r = el.getBoundingClientRect();
    const own = Array.from(el.childNodes).some(n => n.nodeType === 3 && n.textContent.trim());
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.visibility === 'collapse'
        || parseFloat(cs.opacity) < 0.05 || (own && parseFloat(cs.fontSize) < 2)
        || (own && (r.right < 0 || r.bottom < 0 || r.left > 100000 || r.top > 100000))
        || (own && r.width < 1 && r.height < 1) || (own && cs.color === back(el))) gone.push(el);
  }
  gone.forEach(e => e.remove());
  return gone.length;
}"""


def _rendered(url: str, browse: Callable[[], Any]) -> str:
    """The HTML a browser ends up with, scripts run and unseen elements removed, after the
    refusal check."""
    url = check(url)
    with browse() as page:
        page.goto(url, wait_until="load")
        page.evaluate(_UNSEEN_JS)
        return str(page.content())


_SERIAL = itertools.count(1)


def _readable(html: str, url: str) -> tuple[str, str, int]:
    """``(title, text, removed)`` of a page with its hidden content taken out first."""
    cleaned, removed = untrusted.strip_hidden(html)
    title, text = extract(cleaned, url)
    return untrusted.clean_text(title)[0], text, removed


def _labelled(out: dict[str, Any], text: str, url: str, removed: int) -> dict[str, Any]:
    """``out`` with its text cleaned, fenced and marked untrusted; the URLs in it are noted
    as found in fetched content."""
    item = untrusted.untrusted(text, url, next(_SERIAL), html_removed=removed)
    untrusted.shared().page(item.text)
    out.update(text=item.fenced(), untrusted=True, origin=item.origin)
    if item.removed:
        out["hidden_removed"] = item.removed
    return out


def read(url: str, *, limit: int = 6000, fetch: Callable[[str], str] | None = None,
         rendered: bool = False, browse: Callable[[], Any] | None = None) -> dict[str, Any]:
    """One page as text: ``{"url", "title", "text", "rendered"}``, ``"truncated": True`` when cut.

    ``fetch`` is ``url -> html`` (tests pass one; the default is trafilatura's fetcher, or
    urllib). ``rendered=True`` opens the page in a real browser and reads what the scripts
    built; a plain fetch that comes back with fewer than ``THIN`` characters of text falls
    through to that on its own, and back to the plain result when there is no browser.
    ``browse`` is the browser to use, as ``ml_stack.scrape.browser.browser`` gives one.
    Refuses anything ``check`` refuses, before fetching.
    """
    url = check(url)
    if urllib.parse.urlsplit(url).path.lower().endswith(".pdf"):
        return _read_pdf(url, limit)
    fetch = fetch or _fetch
    browse = browse or _browse
    title, text, plain_error, was_rendered, html, hidden = "", "", None, False, "", 0
    if not rendered:
        try:
            html = fetch(url)
            title, text, hidden = _readable(html, url)
        except Exception as exc:  # a 403 to a bot is the commonest reason to render instead
            plain_error = exc
    if rendered or len(text) < THIN:
        try:
            html = _rendered(url, browse)
        except Exception as exc:
            if plain_error is not None:
                raise plain_error from exc
            # no browser, or it failed: the plain read is what there is
        else:
            r_title, r_text, r_hidden = _readable(html, url)
            if r_text or not text:
                title, text, was_rendered, hidden = r_title or title, r_text, True, r_hidden
    elif plain_error is not None:
        raise plain_error
    text, truncated = cut(text, limit)
    out: dict[str, Any] = {"url": url, "title": title, "text": text, "rendered": was_rendered}
    if truncated:
        out["truncated"] = True
    if ahead := next_link(html, url, guess=False):
        out["next"] = ahead
        untrusted.shared().paginate(url, ahead)
    return _labelled(out, text, url, hidden)


def _read_pdf(url: str, limit: int) -> dict[str, Any]:
    """A PDF as text: downloaded into ``downloads_dir()``, with the file's path and hash."""
    got = download(url, downloads_dir(), polite=politeness())
    title, text, pages, hidden = pdftext.visible_text(got.path, limit=limit + 1)
    text, truncated = cut(text, limit)
    out: dict[str, Any] = {"url": url, "title": title, "text": text, "rendered": False,
                           "pdf": str(got.path), "pages": pages, "sha256": got.sha256}
    if truncated:
        out["truncated"] = True
    return _labelled(out, text, url, hidden)


def fetch_file(url: str, kind: str = "pdf", *, licence: str = "") -> Download:
    """Download ``url`` into ``downloads_dir()``; ``kind`` is a key of ``scrape.download.KINDS``
    (pdf, zip, step, kicad, png, jpeg). See ``scrape.download.download`` for what it raises."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(sorted(KINDS))}, not {kind!r}")
    wanted = Wanted(accept=tuple(sorted(KINDS[kind][0])), licence=licence or LICENCE)
    return download(url, downloads_dir(), wanted, politeness())


# Every <img> with a rendered size, largest first, for the page to answer in one round trip.
_IMAGES_JS = """() => Array.from(document.images).map(i => ({
    src: i.currentSrc || i.src || "",
    width: i.naturalWidth || i.width || 0,
    height: i.naturalHeight || i.height || 0}))"""


def look(url: str, *, limit: int = LOOK_CHARS, browse: Callable[[], Any] | None = None,
         fetch_bytes: Callable[[str], bytes] | None = None,
         most: int = MOST_IMAGES) -> dict[str, Any]:
    """A page as a vision model sees it.

    ``{"url", "title", "text", "_images": [png, ...]}`` — a full-page screenshot first,
    then up to ``most`` of the page's largest pictures by rendered area, skipping anything
    under ``LEAST_IMAGE`` square and ``data:`` URIs. ``_images`` is the convention the ask
    loop strips out of a tool result and hands to the model as images; the text is the
    first ``limit`` characters, for a caption. Each picture is fetched through the same
    refusal as the page. Raises ``BrowserUnavailable`` without playwright.
    """
    url = check(url)
    browse = browse or _browse
    fetch_bytes = fetch_bytes or _fetch_bytes
    with browse() as page:
        page.goto(url, wait_until="load")
        html = str(page.content())
        shot = bytes(page.screenshot(full_page=True))
        found = page.evaluate(_IMAGES_JS) or []
    title, text, hidden = _readable(html, url)
    text, _ = cut(text, limit)
    candidates = []
    for item in found:
        if not isinstance(item, Mapping):
            continue
        src = str(item.get("src") or "")
        width, height = int(item.get("width") or 0), int(item.get("height") or 0)
        if not src or src.startswith("data:") or width < LEAST_IMAGE or height < LEAST_IMAGE:
            continue
        candidates.append((width * height, urllib.parse.urljoin(url, src)))
    candidates.sort(key=lambda c: -c[0])
    images: list[bytes] = [shot]
    taken: set[str] = set()
    for _, src in candidates:
        if len(images) > most:
            break
        if src in taken:
            continue
        taken.add(src)
        with contextlib.suppress(Exception):  # a picture that will not come is not the answer
            images.append(bytes(fetch_bytes(src)))
    return _labelled({"url": url, "title": title, "text": text, "_images": images}, text, url,
                     hidden)


# --- the tools ----------------------------------------------------------------------------

# Invented sites under the reserved .example domain: nothing here resolves.
SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for pages about some words, and get back the title, "
                       "link and a line from each. The graph holds this community and "
                       "nothing else; the web is for what the graph cannot know — what a "
                       "company actually does, whether there is a newer release of "
                       "something, what a term means. Answer from the graph first, and "
                       "reach for this only when the graph came back empty or the question "
                       "is about the world outside it. Examples: \"What does Quenlow "
                       "Robotics do?\" → web_search(query=\"Quenlow Robotics\"); \"Is there "
                       "a newer release of the Tessyn compiler?\" → web_search(query="
                       "\"Tessyn compiler latest release\"); \"Who founded Pellard "
                       "Foundry?\" → web_search(query=\"Pellard Foundry founder\"). Do not "
                       "use it for a person or organisation in this community that look_up "
                       "would find, and do not repeat a search with more words when the "
                       "first found nothing — read one of the pages it did find instead.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string",
                      "description": "what to search for, as a few words, e.g. "
                                     "\"Quenlow Robotics\""},
            "page": {"type": "integer",
                     "description": "1 for the first results, 2 for the next, and so on; "
                                    "only when the first page did not hold the answer"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "web_read",
        "description": "Read one web page as text: its title and readable words, cut to a "
                       "few thousand characters. Use it on a link web_search returned, or on "
                       "a link the question itself gives, when a snippet is not enough to "
                       "answer from. Examples: a search returned https://quenlow.example/"
                       "about → web_read(url=\"https://quenlow.example/about\"); \"What does "
                       "this page say? https://pellard.example/news\" → web_read(url="
                       "\"https://pellard.example/news\"); a page that came back nearly "
                       "empty → web_read(url=\"https://tessyn.example\", rendered=true), "
                       "which opens a real browser and is slow, so only then. Do not use it "
                       "for what the graph holds — look_at reads an entry, this reads a "
                       "page — and do not read every result: one or two good pages answer "
                       "most questions.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string",
                    "description": "the page's address, exactly as a search returned it"},
            "rendered": {"type": "boolean",
                         "description": "open it in a browser and read what the scripts "
                                        "built; only when a plain read came back thin"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "web_download",
        "description": "Download a file from a link and keep it on disk, with a record of "
                       "where it came from. Returns the path, the final address, the hash and "
                       "the size. Use it for a datasheet, a 3D model or a footprint a search "
                       "or a page pointed to. Examples: a search returned https://quenlow."
                       "example/ds/arm7.pdf → web_download(url=\"https://quenlow.example/ds/"
                       "arm7.pdf\"); a page links https://pellard.example/models/kiln.step → "
                       "web_download(url=\"https://pellard.example/models/kiln.step\", "
                       "kind=\"step\"). A file whose type or first bytes do not match what "
                       "was asked for is refused, as is anything a site's robots.txt rules "
                       "out. To read a PDF rather than keep it, use web_read.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "the file's address, exactly as given"},
            "kind": {"type": "string",
                     "description": "what the file should be: pdf (the default), step, zip, "
                                    "kicad, png or jpeg"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "web_look",
        "description": "Look at a web page the way a person sees it: a screenshot of the "
                       "whole page and its largest pictures, given to you as images, with "
                       "the first lines of its text. Use it when the layout or a picture is "
                       "what the question is about — a chart, a product photo, what a site "
                       "looks like. Examples: \"What does the Quenlow Robotics site look "
                       "like?\" → web_look(url=\"https://quenlow.example\"); \"Read the "
                       "chart on that page\" → web_look(url=\"https://pellard.example/"
                       "report\"); \"Is their logo blue?\" → web_look(url=\"https://"
                       "tessyn.example\"). For the words on a page use web_read, which is "
                       "faster and holds more text; this is for what words do not carry.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string",
                    "description": "the page's address, exactly as a search returned it"}},
            "required": ["url"]}}},
]

# What a question looks like when it wants each tool — for graph.route's embedder, beside
# ask.TOOL_PROMPTS, and never sent to the chat model. Questions, not descriptions: the
# reason is in graph.route.
PROMPTS: dict[str, tuple[str, ...]] = {
    "web_search": (
        "what does that company actually do?",
        "is there a newer release of this?",
        "search the web for that",
        "what is this term, outside this community?",
        "look it up online",
    ),
    "web_read": (
        "read that page for me",
        "what does this link say?",
        "open the first result and summarise it",
    ),
    "web_download": (
        "download that datasheet",
        "save the pdf from that link",
        "fetch the step file for this part",
    ),
    "web_look": (
        "what does their site look like?",
        "read the chart on that page",
        "show me the picture on that page",
    ),
}


def _schema(name: str) -> dict[str, Any]:
    for schema in SCHEMAS:
        if schema["function"]["name"] == name:
            return schema
    raise KeyError(name)


def tools(*, engine: Engine | None = None, fetch: Callable[[str], str] | None = None,
          browse: Callable[[], Any] | None = None,
          fetch_bytes: Callable[[str], bytes] | None = None,
          vision: bool = False) -> list[tuple[dict[str, Any], Any]]:
    """The web as ``(schema, callable)`` pairs, to pass to ``converse`` beside ``tools_for``.

    ``web_search``, ``web_read`` and ``web_download`` always; ``web_look`` only with
    ``vision=True``. ``engine``, ``fetch``, ``browse`` and ``fetch_bytes`` are the seams
    ``search``, ``read`` and ``look`` take. Each callable takes the parsed arguments and never
    raises: what went wrong comes back as ``{"none": reason}``. A URL a search returned or a
    person typed is fetched; one only fetched content mentioned needs an allow-listed or
    approved host (`ml_stack.net.untrusted.Origins`). Results are marked ``untrusted``.
    """
    seen = untrusted.shared()

    def searching(args: Mapping[str, Any]) -> Any:
        query = str(args.get("query") or "").strip()
        if not query:
            return {"none": "nothing to search for: pass a query"}
        try:
            rows = search(query, engine=engine, page=max(1, int(args.get("page") or 1)))
        except Exception as exc:
            return {"none": f"search unavailable: {exc}"}
        seen.search(rows)
        return rows or {"none": f"Nothing on the web matched {query!r}. Try fewer or "
                                "different words, or answer with what you already have."}

    def reading(args: Mapping[str, Any]) -> Any:
        try:
            seen.admit(str(args.get("url") or ""))
            return read(str(args.get("url") or ""), fetch=fetch, browse=browse,
                        rendered=bool(args.get("rendered")))
        except Exception as exc:
            return {"none": f"could not read {args.get('url')!r}: {exc}"}

    def looking(args: Mapping[str, Any]) -> Any:
        try:
            seen.admit(str(args.get("url") or ""))
            return look(str(args.get("url") or ""), browse=browse, fetch_bytes=fetch_bytes)
        except ImportError as exc:
            return {"none": f"no browser: {exc}"}
        except Exception as exc:
            from ml_stack.scrape.browser import BrowserUnavailable

            why = "no browser" if isinstance(exc, BrowserUnavailable) else \
                f"could not look at {args.get('url')!r}"
            return {"none": f"{why}: {exc}"}

    def downloading(args: Mapping[str, Any]) -> Any:
        try:
            if str(args.get("kind") or "pdf") not in KINDS:
                raise ValueError(f"kind must be one of {', '.join(sorted(KINDS))}")
            seen.admit(str(args.get("url") or ""))
            return as_dict(fetch_file(str(args.get("url") or ""), str(args.get("kind") or "pdf")))
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            return {"none": f"could not download {args.get('url')!r}: {exc}"}

    pairs = [(_schema("web_search"), searching), (_schema("web_read"), reading)]
    if vision:
        pairs.append((_schema("web_look"), looking))
    pairs.append((_schema("web_download"), downloading))
    return pairs


__all__ = ["ENGINES", "PROMPTS", "SCHEMAS", "Refused", "SearchUnavailable", "check", "cut",
           "ddgs_engine", "downloads_dir", "extract", "fetch_file", "look", "politeness", "read",
           "search", "search_pages", "searxng_engine", "tools"]
