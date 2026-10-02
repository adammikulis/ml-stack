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

from ml_stack import net
from ml_stack.http import Refused, ServerError, check

USER_AGENT = "Mozilla/5.0 (compatible; ml-stack)"


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
                ask = net.Ask(purpose="web", admit=False, tries=self.tries,
                              backoff=self.backoff_s,
                              headers={"User-Agent": self.user_agent, "Accept": accept})
                with net.default().open(url, ask) as reply:
                    yield reply
            finally:
                self._last[host] = self.clock()

    @contextlib.contextmanager
    def turn(self, url: str) -> Iterator[str]:
        """Holds this host's turn for a caller that makes the request itself; the URL, once
        the guard and robots.txt have passed."""
        url = self.guard(url)
        if not self.allowed(url):
            raise Disallowed(f"robots.txt asks for {url} not to be fetched")
        host = _host(url)
        with self._lock(host):
            self._pause(host)
            try:
                yield url
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
            ask = net.Ask(purpose="web", admit=False, max_bytes=512 * 1024,
                          headers={"User-Agent": self.user_agent})
            with net.default().open(self.guard(robots_url), ask) as reply:
                parser.parse(reply.read(512 * 1024).decode("utf-8", "replace").splitlines())
        except ServerError as exc:
            if exc.status is not None and 400 <= exc.status < 500:
                self._rules[host] = None
                return None
            parser.parse(["User-agent: *", "Disallow: /"])
        except (OSError, Refused):
            parser.parse(["User-agent: *", "Disallow: /"])
        self._rules[host] = parser
        return parser


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.casefold()
