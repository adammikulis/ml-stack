"""The one place a request to a host outside this machine is made.

`Pipeline.get` and `Pipeline.open` read pages and API answers; `Pipeline.download` (in
``download.py``) brings a file in through staging. Every request is checked against the host
policy, resolved once and connected to the address that was checked, re-checked on every
redirect, bounded in bytes and time, and sent without cookies. HTTPS is verified and a redirect
from HTTPS to HTTP is refused.
"""

from __future__ import annotations

import contextlib
import os
import re
import time
import urllib.parse
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

from ml_stack import http, httpguard
from ml_stack.httpguard import Fetched, Limits, Refused
from ml_stack.net import policy as hosts
from ml_stack.net.hold import Hold, SentinelHold
from ml_stack.net.policy import host_of
from ml_stack.net.scan import Scanner, ScanPolicy
from ml_stack.net.scanners import default_scanners

__all__ = ["ASK", "PAGE", "Ask", "Pipeline", "Reply", "bearer", "default", "use"]

PAGE = Limits(max_bytes=8 * 1024 * 1024, deadline_s=60.0)
"""The limits of a page or an API answer held in memory."""
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class Ask:
    """How one request is made: its ``purpose`` (named in a refusal), extra ``headers``, a bearer
    ``token`` (https only), whether the host policy applies (``admit``), retries, a size cap and ``private`` hosts the person named that may be on this
    network."""

    purpose: str = "fetch"
    headers: dict[str, str] = field(default_factory=dict)
    token: str = ""
    admit: bool = True
    tries: int = 1
    backoff: float = 0.5
    max_bytes: int = 0
    private: tuple[str, ...] = ()


ASK = Ask()


def mirror_hosts() -> frozenset[str]:
    """The host of ``$HF_ENDPOINT``: a mirror the person named may be on this network."""
    named = host_of(os.environ.get("HF_ENDPOINT", ""))
    return frozenset({named}) if named else frozenset()


class Headers(dict[str, str]):
    """Response headers looked up whatever the case of the name."""

    def get(self, name: str, default: Any = None) -> Any:  # type: ignore[override]
        return super().get(name.lower(), default)

    def __getitem__(self, name: str) -> str:
        return super().__getitem__(name.lower())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and super().__contains__(name.lower())

    def get_content_charset(self) -> str | None:
        """The charset of the content type, or None."""
        match = re.search(r"charset=([\w.-]+)", self.get("content-type", ""), re.IGNORECASE)
        return match.group(1) if match else None


class Reply:
    """An open answer: ``status``, ``headers`` (names lower-cased), ``geturl()`` and ``read``."""

    def __init__(self, shown: httpguard.Streaming) -> None:
        self.status, self.headers, self.url = shown.status, Headers(shown.headers), shown.url
        self.redirects = shown.redirects
        self._chunks, self._buffer = shown.chunks, b""

    def geturl(self) -> str:
        """The URL the answer came from, after redirects."""
        return self.url

    def read(self, size: int = -1) -> bytes:
        """Up to ``size`` bytes of the body (all of it for -1)."""
        while size < 0 or len(self._buffer) < size:
            chunk = next(self._chunks, b"")
            if not chunk:
                break
            self._buffer += chunk
        if size < 0:
            out, self._buffer = self._buffer, b""
        else:
            out, self._buffer = self._buffer[:size], self._buffer[size:]
        return out


@dataclass
class Pipeline:
    """Policy, limits, scanners and the hold that every fetch and download goes through."""

    policy: hosts.Policy = field(default_factory=hosts.default)
    limits: Limits = PAGE
    scanners: list[Scanner] = field(default_factory=default_scanners)
    scan_policy: ScanPolicy = field(default_factory=ScanPolicy.load)
    hold: Hold = field(default_factory=SentinelHold)
    sleep: Callable[[float], None] = time.sleep

    def limited(self, purpose: str, *, admit: bool = True, **changes: object) -> Limits:
        """This pipeline's limits with ``changes`` applied and, when ``admit``, the host
        policy checked on every hop."""
        vet = (lambda url: self.policy.admit(url, purpose)) if admit else None
        changes.setdefault("allow_hosts", self.limits.allow_hosts)
        changes["allow_hosts"] = frozenset(changes["allow_hosts"]) | mirror_hosts()  # type: ignore[arg-type]
        return replace(self.limits, vet=vet, **changes)  # type: ignore[arg-type]

    def get(self, url: str, ask: Ask = ASK) -> Fetched:
        """``url``'s answer. `Refused` for a host the policy does not admit, a private
        address at any hop, a downgrade, or a body over the limit."""
        limits = self._limits(ask)
        return httpguard.fetch(url, headers=bearer(url, ask.headers, ask.token, self.plain()), limits=limits)

    def plain(self) -> frozenset[str]:
        """Hosts a token may be sent to without TLS: those named as private-network hosts."""
        return self.limits.allow_hosts | httpguard.allowed_hosts() | mirror_hosts()

    def _limits(self, ask: Ask) -> Limits:
        changes: dict[str, object] = {"max_bytes": ask.max_bytes} if ask.max_bytes else {}
        if ask.private:
            changes["allow_hosts"] = self.limits.allow_hosts | frozenset(ask.private)
        return self.limited(ask.purpose, admit=ask.admit, **changes)

    @contextlib.contextmanager
    def open(self, url: str, ask: Ask = ASK) -> Iterator[Reply]:
        """An answer to read in pieces. A status of 400 or more is `http.ServerError`
        (retried ``tries`` times for 429 and 5xx)."""
        limits, sent, tries = self._limits(ask), bearer(url, ask.headers, ask.token, self.plain()), max(1, ask.tries)
        for attempt in range(tries):
            with httpguard.stream(url, headers=sent, limits=limits) as shown:
                if shown.status < 400:
                    yield Reply(shown)
                    return
                error = http.ServerError(f"{http.shown(url)} -> HTTP {shown.status}",
                                         status=shown.status, headers=shown.headers)
            if shown.status not in RETRY_STATUS or attempt == tries - 1:
                raise error
            self.sleep(ask.backoff * (attempt + 1))

    def json(self, url: str, ask: Ask = ASK) -> object:
        """``url``'s answer parsed as JSON; `http.ServerError` for a 4xx or 5xx."""
        import json

        got = self.get(url, replace(ask, headers={"Accept": "application/json", **ask.headers}))
        if got.status >= 400:
            raise http.ServerError(f"{http.shown(url)} -> HTTP {got.status}", status=got.status,
                                   headers=got.headers)
        try:
            return json.loads(got.body)
        except ValueError as exc:
            raise http.ServerError(f"{http.shown(url)} returned non-JSON") from exc


def bearer(url: str, headers: dict[str, str] | None, token: str,
           plain: frozenset[str] = frozenset()) -> dict[str, str]:
    """``headers`` with ``token`` as a bearer credential; `Refused` unless the URL is https or
    its host is one of ``plain`` (a mirror on this network the person named)."""
    sent = dict(headers or {})
    if token:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https" and (parts.hostname or "") not in plain:
            raise Refused("a token is sent over https only")
        sent["Authorization"] = f"Bearer {token}"
    return sent


_CURRENT: Pipeline | None = None


def default() -> Pipeline:
    """The pipeline in force: the one `use` installed, else a standard one."""
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = Pipeline()
    return _CURRENT


@contextlib.contextmanager
def use(pipeline: Pipeline) -> Iterator[Pipeline]:
    """Make ``pipeline`` the default for the block; a test points the world at a local server."""
    global _CURRENT
    before = _CURRENT
    _CURRENT = pipeline
    try:
        yield pipeline
    finally:
        _CURRENT = before

