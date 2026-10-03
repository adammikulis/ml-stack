"""What the page server checks before it routes a request: the Host name, the Origin of a
POST, its content type and the length it claims."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

__all__ = ["MAX_BODY", "Guarded", "refusal"]

MAX_BODY = 4 * 1024 * 1024
"""The largest request body read, in bytes."""

LOOPBACK = frozenset({"127.0.0.1", "localhost", "[::1]"})


def _host_ok(host: str, port: int) -> bool:
    """True for a loopback name, with this server's port or none."""
    name, sep, tail = host.strip().lower().rpartition(":")
    if not sep or tail.endswith("]"):
        name, tail = host.strip().lower(), ""
    if name not in LOOPBACK:
        return False
    return tail == "" or (tail.isdigit() and int(tail) == port)


def refusal(method: str, headers: dict[str, str], port: int) -> tuple[int, str] | None:
    """The status and message for a request the server will not route, or None.

    ``headers`` is keyed by lower-case name. A POST must come from a loopback origin on this
    port (when it names one), carry ``application/json`` and claim a length that is a
    number no larger than ``MAX_BODY``.
    """
    if not _host_ok(headers.get("host", ""), port):
        return 421, "this server answers to its loopback names only"
    if method != "POST":
        return None
    origin = headers.get("origin")
    if origin is not None:
        parts = urlsplit(origin)
        if parts.scheme != "http" or not _host_ok(parts.netloc, port):
            return 403, "a request from another origin"
    claimed = headers.get("content-length", "0").strip() or "0"
    if not claimed.isascii() or not claimed.isdigit():
        return 400, "Content-Length is not a length"
    if int(claimed) > MAX_BODY:
        return 413, f"a body over {MAX_BODY} bytes"
    if headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        return 400, "a POST carries application/json"
    return None


class Guarded:
    """Mix into an ``http.server`` handler ahead of its routes: ``rejected`` answers a
    request `refusal` rejects."""

    command: str
    headers: Any
    server: Any
    wfile: Any
    close_connection: bool
    send_response: Any
    send_header: Any
    end_headers: Any

    def rejected(self) -> bool:
        """Send the refusal for this request, closing the connection, and say so."""
        found = refusal(self.command, {k.lower(): v for k, v in self.headers.items()},
                        int(self.server.server_address[1]))
        if found is None:
            return False
        code, text = found
        self.close_connection = True
        self.send_response(code)
        blob = text.encode()
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(blob)
        return True
