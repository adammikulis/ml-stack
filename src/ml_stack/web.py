"""The web, as tools a model can call beside the graph's.

The four tools in ``graph.ask`` see the graph and nothing else, which is right for "who
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
  in ``graph.ask``, and the measurement behind it is there.
- **A model must not be able to read the machine it runs on.** ``read`` refuses anything
  that is not http(s), and any host that resolves to a loopback, private or link-local
  address, before a byte is fetched or a browser navigates. The check is on what the name
  *resolves to*, not what it looks like, because ``localhost`` is spelt many ways.
- **The examples in the descriptions are the point.** Measured in ``graph.ask``: a worked
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
import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

Engine = Callable[[str, int], list[dict[str, Any]]]
"""``(query, limit) -> [{"title", "url", "snippet"}, ...]``; may raise SearchUnavailable."""

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
PROFILE = Path(os.environ.get("MLSTACK_WEB_PROFILE") or "~/.ml-stack/web")


class SearchUnavailable(RuntimeError):
    """The search engine would not answer: rate-limited, timed out, offline, or missing."""


class Refused(ValueError):
    """A URL this module will not fetch: not http(s), or a host on this machine's side."""


# --- search -------------------------------------------------------------------------------


def ddgs_engine(query: str, limit: int) -> list[dict[str, Any]]:
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
        rows = DDGS().text(query, max_results=limit, backend=backend)
    except DDGSException as exc:
        raise SearchUnavailable(f"{type(exc).__name__}: {exc}") from exc
    return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in rows]


def searxng_engine(query: str, limit: int) -> list[dict[str, Any]]:
    """Search a SearXNG instance at ``SEARXNG_URL`` through its ``/search?format=json``.

    Self-hosted, so nobody rate-limits it but its owner; the JSON format has to be enabled
    in the instance's ``settings.yml``. Stdlib only.
    """
    base = (os.environ.get("SEARXNG_URL") or "").rstrip("/")
    if not base:
        raise SearchUnavailable("SEARXNG_URL is not set")
    url = f"{base}/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
    try:
        body = _http(url, accept="application/json")
        payload = json.loads(body.decode("utf-8", "replace"))
    except (OSError, ValueError) as exc:
        raise SearchUnavailable(f"searxng at {base}: {exc}") from exc
    rows = payload.get("results") if isinstance(payload, Mapping) else None
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in (rows or [])[:limit] if isinstance(r, Mapping)]


ENGINES: dict[str, Engine] = {"ddgs": ddgs_engine, "searxng": searxng_engine}


def search(query: str, *, limit: int = 8, engine: Engine | None = None) -> list[dict[str, Any]]:
    """Pages about ``query``: ``[{"title", "url", "snippet"}, ...]``, at most ``limit``.

    ``engine`` is ``(query, limit) -> rows``; when None, ``ENGINES[MLSTACK_SEARCH]`` with
    ``ddgs`` the default. A blank query finds nothing. Raises ``SearchUnavailable`` when
    the engine would not answer, and ``ImportError`` when it is not installed.
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
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in engine(wanted, limit):
        url = str(row.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({"title": " ".join(str(row.get("title") or "").split()),
                    "url": url,
                    "snippet": " ".join(str(row.get("snippet") or "").split())})
        if len(out) >= limit:
            break
    return out


# --- fetching, and what may be fetched ----------------------------------------------------


def _addresses(host: str) -> list[str]:
    """Every address a host name resolves to. Separate so a test can answer for DNS."""
    try:
        return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})
    except socket.gaierror as exc:
        raise Refused(f"cannot resolve {host!r}: {exc}") from exc


def check(url: str) -> str:
    """The URL, if it may be fetched; ``Refused`` otherwise.

    http(s) only, and only to a host whose every address is a public one. ``file:``,
    ``localhost``, ``127.0.0.0/8``, ``10.0.0.0/8``, ``192.168.0.0/16``, ``172.16.0.0/12``,
    link-local and the IPv6 equivalents are all refused, by what the name resolves to.
    """
    parts = urllib.parse.urlsplit((url or "").strip())
    if parts.scheme not in ("http", "https"):
        raise Refused(f"only http(s) is read, not {parts.scheme or 'a bare path'}: {url!r}")
    host = (parts.hostname or "").strip("[]").casefold()
    if not host:
        raise Refused(f"no host in {url!r}")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise Refused(f"{host} is this machine")
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        addresses = _addresses(host)
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if not ip.is_global:
            raise Refused(f"{host} resolves to {ip}, which is not on the public internet")
    return urllib.parse.urlunsplit(parts)


def _http(url: str, *, accept: str = "*/*", most: int = MOST_BYTES) -> bytes:
    """One GET with a size cap and no refusal: for a search backend, which is often on
    this side of the router. Pages a model chose go through ``_get``."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as reply:  # noqa: S310
        return reply.read(most)


def _get(url: str, *, accept: str = "*/*", most: int = MOST_BYTES) -> bytes:
    """One GET, with a size cap, after ``check``."""
    return _http(check(url), accept=accept, most=most)


def _fetch(url: str) -> str:
    """A page's HTML: trafilatura's fetcher when installed, urllib when not."""
    url = check(url)
    try:
        import trafilatura
    except ImportError:
        trafilatura = None
    if trafilatura is not None:
        html = trafilatura.fetch_url(url)
        if html:
            return html
        # trafilatura swallows the reason; urllib will say what it was
    body = _get(url, accept="text/html,application/xhtml+xml,*/*;q=0.5")
    return body.decode("utf-8", "replace")


def _fetch_bytes(url: str) -> bytes:
    """A picture's bytes, after the same check as a page."""
    return _get(url, accept="image/*")


class _Stripper(HTMLParser):
    """The words of a page, without its scripts, styles or tags. The fallback reader."""

    SKIP = {"script", "style", "noscript", "template", "svg"}
    BREAK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section",
             "article", "header", "footer", "nav", "blockquote", "pre"}

    def __init__(self) -> None:
        super().__init__()
        self.title: list[str] = []
        self.parts: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title.append(data)
        elif not self._skip:
            self.parts.append(data)


def extract(html: str, url: str = "") -> tuple[str, str]:
    """``(title, text)`` read out of a page: trafilatura when installed, tags stripped when not."""
    try:
        import trafilatura
    except ImportError:
        trafilatura = None
    title, text = "", ""
    if trafilatura is not None:
        with contextlib.suppress(Exception):
            doc = trafilatura.bare_extraction(html, url=url or None, with_metadata=True)
            if doc is not None:
                title, text = str(doc.title or ""), str(doc.text or "")
    if not text:
        stripper = _Stripper()
        with contextlib.suppress(Exception):
            stripper.feed(html)
        title = title or " ".join(unescape("".join(stripper.title)).split())
        lines = [" ".join(unescape(line).split()) for line in "".join(stripper.parts).split("\n")]
        text = "\n".join(line for line in lines if line)
    return title.strip(), text.strip()


_SENTENCE_END = re.compile(r"[.!?…]['\")\]]?(?=\s)|\n")


def cut(text: str, limit: int) -> tuple[str, bool]:
    """``text`` no longer than ``limit``, ended at a sentence when one is near enough.

    A page cut mid-word reads as broken; a page cut mid-sentence reads as a claim that was
    never made. So the cut goes back to the last sentence end in the second half of the
    window, then to the last space, and only then to the character.
    """
    if len(text) <= limit:
        return text, False
    window = text[:limit]
    ends = [m.end() for m in _SENTENCE_END.finditer(window)]
    at = ends[-1] if ends and ends[-1] >= limit // 2 else 0
    if not at:
        space = window.rfind(" ")
        at = space if space >= limit // 2 else limit
    return window[:at].rstrip(), True


# --- reading ------------------------------------------------------------------------------


def _browse() -> Any:
    """A page in a real browser, as a context manager; ``BrowserUnavailable`` without playwright."""
    from ml_stack.scrape.browser import Window, browser

    return browser(Window(profile=PROFILE))


def _rendered(url: str, browse: Callable[[], Any]) -> str:
    """The HTML a browser ends up with, scripts run, after the refusal check."""
    url = check(url)
    with browse() as page:
        page.goto(url, wait_until="load")
        return str(page.content())


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
    fetch = fetch or _fetch
    browse = browse or _browse
    title, text, plain_error, was_rendered = "", "", None, False
    if not rendered:
        try:
            title, text = extract(fetch(url), url)
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
            r_title, r_text = extract(html, url)
            if r_text or not text:
                title, text, was_rendered = r_title or title, r_text, True
    elif plain_error is not None:
        raise plain_error
    text, truncated = cut(text, limit)
    out: dict[str, Any] = {"url": url, "title": title, "text": text, "rendered": was_rendered}
    if truncated:
        out["truncated"] = True
    return out


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
    title, text = extract(html, url)
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
    return {"url": url, "title": title, "text": text, "_images": images}


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
                                     "\"Quenlow Robotics\""}},
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

    ``web_search`` and ``web_read`` always; ``web_look`` only with ``vision=True``, because a
    model that cannot see gains nothing from a screenshot and pays for the description.
    ``engine``, ``fetch``, ``browse`` and ``fetch_bytes`` are the seams ``search``, ``read``
    and ``look`` take, for a test or a project with its own transport. Each callable takes
    the parsed arguments mapping and never raises: what went wrong comes back as
    ``{"none": reason}``, which a model reads as "move on", where an exception ends the turn.
    """
    def searching(args: Mapping[str, Any]) -> Any:
        query = str(args.get("query") or "").strip()
        if not query:
            return {"none": "nothing to search for: pass a query"}
        try:
            rows = search(query, engine=engine)
        except Exception as exc:
            return {"none": f"search unavailable: {exc}"}
        return rows or {"none": f"Nothing on the web matched {query!r}. Try fewer or "
                                "different words, or answer with what you already have."}

    def reading(args: Mapping[str, Any]) -> Any:
        try:
            return read(str(args.get("url") or ""), fetch=fetch, browse=browse,
                        rendered=bool(args.get("rendered")))
        except Exception as exc:
            return {"none": f"could not read {args.get('url')!r}: {exc}"}

    def looking(args: Mapping[str, Any]) -> Any:
        try:
            return look(str(args.get("url") or ""), browse=browse, fetch_bytes=fetch_bytes)
        except ImportError as exc:
            return {"none": f"no browser: {exc}"}
        except Exception as exc:
            from ml_stack.scrape.browser import BrowserUnavailable

            why = "no browser" if isinstance(exc, BrowserUnavailable) else \
                f"could not look at {args.get('url')!r}"
            return {"none": f"{why}: {exc}"}

    pairs = [(_schema("web_search"), searching), (_schema("web_read"), reading)]
    if vision:
        pairs.append((_schema("web_look"), looking))
    return pairs


__all__ = ["ENGINES", "PROMPTS", "SCHEMAS", "Refused", "SearchUnavailable", "check", "cut",
           "ddgs_engine", "extract", "look", "read", "search", "searxng_engine", "tools"]
