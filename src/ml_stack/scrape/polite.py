"""Fetching a site the way a considerate client does: robots.txt, a gap between requests, one
request per host at a time, and retries with backoff."""

from __future__ import annotations

import contextlib
import threading
import time
import urllib.parse
import urllib.robotparser
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ml_stack.http import Refused, Retry, ServerError, check, open_stream

USER_AGENT = "Mozilla/5.0 (compatible; ml-stack)"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
ROBOTS_TIMEOUT_S = 10.0


class Disallowed(Refused):
    """The site's robots.txt asks clients not to fetch this URL."""


@dataclass
class Polite:
    """Rules for talking to hosts: ``allowed`` reads robots.txt, ``fetch`` spaces requests.

    ``min_interval_s`` is the least gap between two requests to one host; a robots.txt
    ``Crawl-delay`` longer than that wins. ``robots=False`` stops consulting robots.txt.
    ``guard`` vets every URL and every redirect target; ``check`` refuses private hosts.
    """

    robots: bool = True
    min_interval_s: float = 1.0
    user_agent: str = USER_AGENT
    tries: int = 3
    backoff_s: float = 1.0
    timeout_s: float = 30.0
    guard: Callable[[str], str] = check
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _rules: dict[str, urllib.robotparser.RobotFileParser | None] = field(default_factory=dict)
    _locks: dict[str, threading.Lock] = field(default_factory=dict)
    _last: dict[str, float] = field(default_factory=dict)
    _mutex: threading.Lock = field(default_factory=threading.Lock)

    def allowed(self, url: str) -> bool:
        """Whether robots.txt lets this client fetch ``url``; always True with ``robots=False``."""
        if not self.robots:
            return True
        rules = self._robots_for(url)
        return rules is None or rules.can_fetch(self.user_agent, url)

    def note(self, url: str) -> str:
        """One line saying how robots.txt was treated for ``url``, for a provenance record."""
        if not self.robots:
            return "robots.txt not consulted"
        rules = self._robots_for(url)
        if rules is None:
            return "robots.txt absent or empty: no restrictions"
        return "robots.txt allows" if rules.can_fetch(self.user_agent, url) \
            else "robots.txt disallows"

    @contextlib.contextmanager
    def fetch(self, url: str, *, accept: str = "*/*") -> Iterator[Any]:
        """The open response for ``url``, holding this host's turn until the block ends.

        Raises ``Disallowed`` when robots.txt says no, ``Refused`` from the guard, and
        ``ServerError`` once the retries are spent.
        """
        url = self.guard(url)
        if not self.allowed(url):
            raise Disallowed(f"robots.txt asks for {url} not to be fetched")
        host = _host(url)
        with self._lock(host):
            self._pause(host)
            try:
                with open_stream(
                        url, headers={"User-Agent": self.user_agent, "Accept": accept},
                        timeout=self.timeout_s, guard=self.guard,
                        retry=Retry(tries=self.tries, backoff=self.backoff_s,
                                    on_status=RETRY_STATUS)) as reply:
                    yield reply
            finally:
                self._last[host] = self.clock()

    def _lock(self, host: str) -> threading.Lock:
        with self._mutex:
            return self._locks.setdefault(host, threading.Lock())

    def _pause(self, host: str) -> None:
        gap = self.min_interval_s
        rules = self._rules.get(host)
        if rules is not None:
            gap = max(gap, float(rules.crawl_delay(self.user_agent) or 0))
        last = self._last.get(host)
        if last is not None and (wait := last + gap - self.clock()) > 0:
            self.sleep(wait)

    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        host = _host(url)
        if host in self._rules:
            return self._rules[host]
        parts = urllib.parse.urlsplit(url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        try:
            with open_stream(self.guard(robots_url), timeout=ROBOTS_TIMEOUT_S,
                             headers={"User-Agent": self.user_agent}, guard=self.guard) as reply:
                parser.parse(reply.read(512 * 1024).decode("utf-8", "replace").splitlines())
        except ServerError as exc:
            if exc.status is not None and 400 <= exc.status < 500:
                self._rules[host] = None
                return None
            parser.parse(["User-agent: *", "Disallow: /"])
        except OSError:
            parser.parse(["User-agent: *", "Disallow: /"])
        self._rules[host] = parser
        return parser


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.casefold()
