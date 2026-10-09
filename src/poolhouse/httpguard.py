"""Fetching a URL somebody else chose: public addresses only, every hop checked, bounded.

`fetch` resolves the host once, refuses any address that is not on the public internet,
connects to that address (so a second DNS answer cannot redirect it), and follows redirects
by hand, checking each hop the same way. The body is capped in bytes and in time, and a
compressed body is capped after it is expanded.
"""

from __future__ import annotations

import http.client
import ipaddress
import os
import socket
import ssl
import time
import urllib.parse
import zlib
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from poolhouse.files import writing

__all__ = [
    "DEFAULT",
    "DOWNLOAD",
    "MOST_BYTES",
    "Fetched",
    "Limits",
    "Refused",
    "Streaming",
    "TooLarge",
    "allowed_address",
    "allowed_hosts",
    "download",
    "fetch",
    "resolve",
    "split",
    "stream",
]

MOST_BYTES = 8 * 1024 * 1024
MOST_REDIRECTS = 5
CHUNK = 64 * 1024
MOST_RATIO = 200
REDIRECTS = {301, 302, 303, 307, 308}
USER_AGENT = "poolhouse"
SECRET_HEADERS = {"authorization", "cookie", "proxy-authorization", "x-api-key"}
NAT64 = ipaddress.ip_network("64:ff9b::/96")

Resolver = Callable[[str, int], Iterable[str]]
"""``(host, port) -> addresses``; a test answers for DNS with one."""


class Refused(ValueError):
    """A URL that will not be fetched, or a fetch that broke one of its limits."""


class TooLarge(Refused):
    """A body, once expanded, over the cap."""


@dataclass(frozen=True, slots=True)
class Limits:
    """What a fetch may spend: bytes, redirects, and time (``timeout`` per socket wait,
    ``deadline_s`` for the whole). ``allow_hosts`` names hosts, by name or address, that may
    be private because the caller means to reach them (a peer on the LAN); a redirect to any
    other host is checked as usual. ``resolver`` answers for DNS. ``vet`` is called with every
    URL before it is connected to, the first and each redirect target, and raises to refuse."""

    max_bytes: int = MOST_BYTES
    max_redirects: int = MOST_REDIRECTS
    timeout: float = 20.0
    deadline_s: float = 60.0
    allow_hosts: frozenset[str] = frozenset()
    resolver: Resolver | None = None
    context: ssl.SSLContext | None = None
    vet: Callable[[str], None] | None = None


DEFAULT = Limits()
DOWNLOAD = Limits(max_bytes=1 << 30, deadline_s=3600.0)
"""The limits of a file saved to disk."""


@dataclass(frozen=True, slots=True)
class Trip:
    """One fetch under way: its limits and the moment it must be over."""

    limits: Limits
    deadline: float

    @classmethod
    def begin(cls, limits: Limits) -> Trip:
        return cls(limits, time.monotonic() + limits.deadline_s)

    def left(self) -> float:
        """Seconds until the deadline; `Refused` once it has passed."""
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise Refused("the fetch ran out of time")
        return remaining


@dataclass(frozen=True, slots=True)
class Fetched:
    """The final answer: where it came from, its status, headers and body."""

    url: str
    status: int
    headers: dict[str, str]
    body: bytes
    redirects: tuple[str, ...] = field(default=())


ALLOW_ENV = "POOLHOUSE_FETCH_ALLOW_HOSTS"


def allowed_hosts() -> frozenset[str]:
    """Hosts the operator has named, comma separated in ``POOLHOUSE_FETCH_ALLOW_HOSTS``, as
    ones that may be private (a mirror on the LAN). Nothing is allowed by default."""
    return frozenset(one.strip().lower() for one in os.environ.get(ALLOW_ENV, "").split(",")
                     if one.strip())


def allowed_address(address: str) -> bool:
    """Whether ``address`` is on the public internet: not loopback, private, link-local
    (the cloud metadata address included), carrier-grade NAT, multicast, reserved or
    unspecified, and not one of those wrapped in IPv6."""
    ip = ipaddress.ip_address(address.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return allowed_address(str(ip.ipv4_mapped))
        if ip in NAT64:
            return allowed_address(str(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)))
        if ip.sixtofour is not None:
            return allowed_address(str(ip.sixtofour))
    return ip.is_global and not (ip.is_multicast or ip.is_reserved or ip.is_unspecified
                                 or ip.is_loopback or ip.is_link_local)


def _system_resolver(host: str, port: int) -> list[str]:
    try:
        return sorted({info[4][0] for info in socket.getaddrinfo(host, port,
                                                                 type=socket.SOCK_STREAM)})
    except socket.gaierror as exc:
        raise Refused(f"cannot resolve {host!r}: {exc}") from None


def resolve(host: str, port: int, limits: Limits = DEFAULT) -> list[str]:
    """Every address ``host`` names; `Refused` when any is not public (unless ``host`` is one
    of the limits' ``allow_hosts``)."""
    named = limits.allow_hosts | allowed_hosts()
    allow_private, resolver = host in named or f"{host}:{port}" in named, limits.resolver
    if not allow_private and (host == "localhost"
                              or host.endswith((".localhost", ".local", ".internal"))):
        raise Refused(f"{host} is this machine or its network")
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        addresses = list((resolver or _system_resolver)(host, port))
    if not addresses:
        raise Refused(f"{host} resolves to nothing")
    for address in addresses:
        if not allow_private and not allowed_address(address):
            raise Refused(f"{host} resolves to {address.split('%')[0]}, "
                          "which is not on the public internet")
    return addresses


def split(url: str) -> tuple[urllib.parse.SplitResult, str, int]:
    if not isinstance(url, str) or any(ord(ch) < 33 or ord(ch) == 127 for ch in url.strip()):
        raise Refused("a URL has no spaces or control characters")
    parts = urllib.parse.urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise Refused(f"only http(s) is fetched, not {parts.scheme or 'a bare path'}")
    try:
        host = (parts.hostname or "").encode("idna").decode("ascii")
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except (UnicodeError, ValueError):
        raise Refused("that URL has no usable host or port") from None
    if not host:
        raise Refused("that URL has no host")
    if parts.username or parts.password:
        raise Refused("a URL does not carry a password")
    return parts, host, port


class _Pinned(http.client.HTTPConnection):
    """Connects to the address already checked, whatever the name resolves to now."""

    def __init__(self, host: str, port: int, address: str, timeout: float,
                 context: ssl.SSLContext | None) -> None:
        super().__init__(host, port, timeout=timeout)
        self.address, self.context = address, context

    def connect(self) -> None:
        self.sock = socket.create_connection((self.address, self.port), self.timeout)
        if self.context is not None:
            self.sock = self.context.wrap_socket(self.sock, server_hostname=self.host)


def _expanded(chunks: Iterator[bytes], encoding: str, most: int) -> Iterator[bytes]:
    """The body, undone from gzip or deflate, refusing a bomb: over ``most`` bytes out, or
    over `MOST_RATIO` times the bytes in."""
    if encoding in ("", "identity"):
        yield from chunks
        return
    if encoding not in ("gzip", "x-gzip", "deflate"):
        raise Refused(f"cannot read a {encoding} body")
    inflater = zlib.decompressobj(47)
    taken = given = 0
    for chunk in chunks:
        taken += len(chunk)
        data = chunk
        while data:
            out = inflater.decompress(data, CHUNK)
            data = inflater.unconsumed_tail
            given += len(out)
            if given > most:
                raise TooLarge("the body expands past its limit")
            if given > MOST_RATIO * taken + CHUNK:
                raise TooLarge(f"the body expands more than {MOST_RATIO} times over")
            if out:
                yield out
    tail = inflater.flush()
    if given + len(tail) > most:
        raise TooLarge("the body expands past its limit")
    if tail:
        yield tail


def _chunks(response: http.client.HTTPResponse, trip: Trip) -> Iterator[bytes]:
    while True:
        trip.left()
        data = response.read1(CHUNK)
        if not data:
            return
        yield data


def _hop(url: str, method: str, data: bytes | None, headers: dict[str, str],
         trip: Trip) -> tuple[http.client.HTTPResponse, _Pinned, urllib.parse.SplitResult]:
    limits = trip.limits
    parts, host, port = split(url)
    if limits.vet is not None:
        limits.vet(url)
    addresses = resolve(host, port, limits)
    tls = (limits.context or ssl.create_default_context()) if parts.scheme == "https" else None
    last: OSError | None = None
    for address in addresses:
        conn = _Pinned(host, port, address, min(limits.timeout, trip.left()), tls)
        try:
            target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
            conn.request(method, target, body=data, headers=headers)
            return conn.getresponse(), conn, parts
        except OSError as exc:
            conn.close()
            last = exc
    raise Refused(f"cannot reach {host}: {last}")


def _open(url: str, method: str, data: bytes | None, sent: dict[str, str], trip: Trip) -> tuple[
        http.client.HTTPResponse, http.client.HTTPConnection, str, tuple[str, ...]]:
    """The first answer that is not a redirect, with its connection, final URL and the hops."""
    seen: list[str] = []
    for _ in range(trip.limits.max_redirects + 1):
        response, conn, parts = _hop(url, method, data, sent, trip)
        location = response.getheader("Location")
        if response.status not in REDIRECTS or not location:
            return response, conn, url, tuple(seen)
        conn.close()
        seen.append(url)
        nxt = urllib.parse.urljoin(url, location)
        if parts.scheme == "https" and urllib.parse.urlsplit(nxt).scheme != "https":
            raise Refused("a redirect from https to plain http is refused")
        if urllib.parse.urlsplit(nxt).netloc.lower() != parts.netloc.lower():
            sent = {k: v for k, v in sent.items() if k.lower() not in SECRET_HEADERS}
        if response.status == 303 or (response.status in (301, 302)
                                      and method not in ("GET", "HEAD")):
            method, data = "GET", None
            sent = {k: v for k, v in sent.items() if k.lower() != "content-type"}
        url = nxt
    raise Refused(f"more than {trip.limits.max_redirects} redirects")


def _body(response: http.client.HTTPResponse, trip: Trip) -> Iterator[bytes]:
    most = trip.limits.max_bytes
    declared = response.getheader("Content-Length")
    encoding = (response.getheader("Content-Encoding") or "").strip().lower()
    if declared and declared.isdigit() and int(declared) > most and not encoding:
        raise TooLarge(f"the answer is {declared} bytes, over {most}")
    total = 0
    for piece in _expanded(_chunks(response, trip), encoding, most):
        total += len(piece)
        if total > most:
            raise TooLarge(f"the answer is over {most} bytes")
        yield piece


def _request(headers: dict[str, str] | None, encoding: str = "gzip") -> dict[str, str]:
    sent = {"User-Agent": USER_AGENT, "Accept-Encoding": encoding, **(headers or {})}
    for name, value in sent.items():
        if "\r" in f"{name}{value}" or "\n" in f"{name}{value}":
            raise Refused("a header holds a line break")
    return sent


def fetch(url: str, *, method: str = "GET", data: bytes | None = None,
          headers: dict[str, str] | None = None, limits: Limits = DEFAULT) -> Fetched:
    """``url``'s answer as `Fetched`, or `Refused` for anything this module will not do.

    ``limits.timeout`` bounds each socket wait and ``limits.deadline_s`` the whole fetch, so
    a server that sends one byte a second ends at the deadline.
    """
    trip = Trip.begin(limits)
    response, conn, final, seen = _open(url, method, data, _request(headers), trip)
    try:
        body = b"".join(_body(response, trip))
        return Fetched(final, response.status, {k.lower(): v for k, v in response.getheaders()},
                       body, seen)
    finally:
        conn.close()


@dataclass(frozen=True, slots=True)
class Streaming:
    """An answer being read: its status and headers now, its body as `chunks`."""

    url: str
    status: int
    headers: dict[str, str]
    chunks: Iterator[bytes]
    redirects: tuple[str, ...] = ()


@contextmanager
def stream(url: str, *, headers: dict[str, str] | None = None,
           limits: Limits = DOWNLOAD) -> Iterator[Streaming]:
    """``url`` opened for reading in pieces, under the same checks as `fetch`.

    The status is not judged here: a caller resuming a download reads a 206 or a 416 itself.
    The body is asked for uncompressed, so a byte range means bytes of the file.
    """
    trip = Trip.begin(limits)
    response, conn, final, seen = _open(url, "GET", None, _request(headers, "identity"), trip)
    try:
        yield Streaming(final, response.status,
                        {k.lower(): v for k, v in response.getheaders()},
                        _body(response, trip), seen)
    finally:
        conn.close()


def download(url: str, dest: Path | str, *, headers: dict[str, str] | None = None,
             limits: Limits = DOWNLOAD) -> Path:
    """``url``'s body streamed to ``dest`` in one step; nothing is left behind when it fails."""
    trip = Trip.begin(limits)
    response, conn, _, _ = _open(url, "GET", None, _request(headers), trip)
    try:
        if response.status != 200:
            raise Refused(f"{url} answered {response.status}")
        with writing(Path(dest)) as draft, draft.open("wb") as out:
            for piece in _body(response, trip):
                out.write(piece)
    finally:
        conn.close()
    return Path(dest)
