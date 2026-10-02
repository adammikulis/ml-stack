"""What a request is allowed to claim about its own size and speed.

A peer-supplied ``Content-Length``, ``Range`` or ``Content-Range`` is a claim, not a fact:
`declared_length` and `content_range` refuse the ones that are malformed, negative, absurdly
large or inconsistent, `read_body` reads exactly what was declared within a deadline, and
`LimitedServer` caps how many connections are open and how long a client may take to send
its headers.
"""

from __future__ import annotations

import contextlib
import ipaddress
import re
import socket
import ssl
import threading
import time
from typing import Any

from ml_stack.http import Server

__all__ = ["HEADER_S", "MOST_BODY", "MOST_CONNECTIONS", "MOST_UPLOAD", "Limited",
           "LimitedServer", "Malformed", "addressed_to_this_machine", "content_range", "declared_length", "read_body",
           "requested_range"]

MOST_BODY = 32 << 20
MOST_UPLOAD = 24 << 20
MOST_CONNECTIONS = 64
HEADER_S = 15.0
SOCKET_S = 30.0
HANDSHAKE_S = 10.0
SLOWEST_BYTES_PER_S = 16 * 1024
DIGITS = re.compile(r"[0-9]{1,18}")
RANGE = re.compile(r"bytes (?P<start>[0-9]{1,18})-(?P<end>[0-9]{1,18})/(?P<total>[0-9]{1,18}|\*)")
SPAN = re.compile(r"bytes=(?P<start>[0-9]{1,18})-(?P<end>[0-9]{0,18})")


class Malformed(Exception):
    """A request whose framing is refused: the HTTP status and what to tell the client."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


def declared_length(headers: Any, most: int = MOST_BODY) -> int:
    """The ``Content-Length`` a request declares, 0 when it declares none.

    Refused with 400 unless it is one run of digits (no sign, no spaces, no repeats that
    disagree) and with 413 above ``most``.
    """
    seen = {value.strip() for value in headers.get_all("Content-Length") or []}
    if not seen:
        return 0
    if len(seen) > 1 or not DIGITS.fullmatch(next(iter(seen))):
        raise Malformed(400, "Content-Length is not a single non-negative number")
    length = int(next(iter(seen)))
    if length > most:
        raise Malformed(413, f"the body is {length} bytes; the most taken is {most}")
    return length


def read_body(handler: Any, most: int = MOST_BODY) -> bytes:
    """Exactly the bytes the request declares, or `Malformed` for a lie, a stall or an excess."""
    length = declared_length(handler.headers, most)
    deadline = time.monotonic() + 30.0 + length / SLOWEST_BYTES_PER_S
    out = bytearray()
    while len(out) < length:
        if time.monotonic() > deadline:
            raise Malformed(408, "the body arrived too slowly")
        try:
            piece = handler.rfile.read(min(1 << 20, length - len(out)))
        except (TimeoutError, OSError):
            raise Malformed(408, "the body stopped arriving") from None
        if not piece:
            raise Malformed(400, f"the body ended at {len(out)} of {length} bytes")
        out += piece
    return bytes(out)


def content_range(header: str, length: int) -> tuple[int, int | None]:
    """``(offset, total)`` from a PUT's ``Content-Range``; ``(0, None)`` when it has none.

    Refused with 400 unless it is ``bytes a-b/total`` with a <= b, b < total and a span
    equal to the body's ``length``.
    """
    if not header:
        return 0, None
    got = RANGE.fullmatch(header.strip())
    if got is None:
        raise Malformed(400, "Content-Range is not 'bytes START-END/TOTAL'")
    start, end = int(got["start"]), int(got["end"])
    total = None if got["total"] == "*" else int(got["total"])
    if end < start or (total is not None and end >= total) or end - start + 1 != length:
        raise Malformed(400, "Content-Range disagrees with the body it came with")
    return start, total


def requested_range(header: str) -> tuple[int, int | None]:
    """``(start, end)`` from a GET's ``Range`` (first span only); ``(0, None)`` for none.

    Refused with 400 for a suffix range, an unreadable one, or an end before its start.
    """
    if not header:
        return 0, None
    got = SPAN.fullmatch(header.split(",")[0].strip())
    if got is None:
        if header.startswith("bytes=-"):
            raise Malformed(400, "suffix ranges are not served")
        raise Malformed(400, "only 'bytes=START-END' ranges are served")
    start = int(got["start"])
    end = int(got["end"]) if got["end"] else None
    if end is not None and end < start:
        raise Malformed(400, "the range ends before it starts")
    return start, end


def addressed_to_this_machine(host_header: str, client: str) -> bool:
    """Whether a request came from this machine and names it by address, not by a DNS name.

    A web page can make a browser send a request to loopback, but only under a name an
    attacker's DNS controls; a request from loopback addressed to an IP literal or
    ``localhost`` was typed by a person or a local program."""
    host = host_header.rsplit(":", 1)[0].strip("[]").lower() if not host_header.startswith("[") \
        else host_header.split("]")[0].strip("[").lower()
    try:
        ipaddress.ip_address(host)
        named_by_address = True
    except ValueError:
        named_by_address = host == "localhost"
    try:
        loopback = ipaddress.ip_address(client.split("%")[0]).is_loopback
    except ValueError:
        loopback = False
    return loopback and named_by_address


class Limited:
    """Handler mixin: a socket timeout, and a deadline for the request line and headers."""

    timeout = SOCKET_S
    header_s = HEADER_S

    def setup(self) -> None:
        """On a TLS server, shake hands with a client that starts with a TLS hello; a client
        from this machine may speak plain HTTP, any other that does is dropped."""
        context = getattr(self.server, "tls", None)  # type: ignore[attr-defined]
        if context is not None:
            sock = self.request  # type: ignore[attr-defined]
            sock.settimeout(HANDSHAKE_S)
            if sock.recv(1, socket.MSG_PEEK) == b"\x16":
                self.request = context.wrap_socket(sock, server_side=True)
            elif not addressed_to_this_machine("localhost", self.client_address[0]):  # type: ignore[attr-defined]
                raise ConnectionAbortedError("plain HTTP from another machine")
        super().setup()  # type: ignore[misc]

    def handle_one_request(self) -> None:
        guard = threading.Timer(self.header_s, self._hang_up)
        guard.daemon = True
        self._header_timer = guard
        guard.start()
        try:
            super().handle_one_request()  # type: ignore[misc]
        finally:
            guard.cancel()

    def parse_request(self) -> bool:
        got = super().parse_request()  # type: ignore[misc]
        self._header_timer.cancel()
        return bool(got)

    def _hang_up(self) -> None:
        with contextlib.suppress(OSError):
            self.connection.shutdown(socket.SHUT_RDWR)  # type: ignore[attr-defined]


class LimitedServer(Server):
    """A threaded server that answers 503 to a connection past ``most`` open at once."""

    def __init__(self, address: tuple[str, int], handler: Any, *, most: int = MOST_CONNECTIONS,
                 tls: ssl.SSLContext | None = None) -> None:
        super().__init__(address, handler)
        self._room = threading.BoundedSemaphore(most)
        self.tls = tls
        """With a context, a client that speaks TLS is served over it; see `Limited.setup`."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A handshake that failed or a client that hung up is not worth a traceback."""
        import sys

        if not isinstance(sys.exc_info()[1], (ssl.SSLError, ConnectionError, TimeoutError)):
            super().handle_error(request, client_address)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._room.acquire(blocking=False):
            with contextlib.suppress(OSError):
                request.sendall(b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
                request.settimeout(0.2)
                request.recv(65536)
            self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._room.release()
