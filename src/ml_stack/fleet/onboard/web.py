"""The one HTTP handler the onboarding servers share.

Pairing, file sharing and the bootstrap offer each say what a request means as a function
from `Call` to `Reply`; this module is the only place that speaks HTTP for them. It reads a
bounded body, writes the reply (a stream for files) and turns a framing error into a status.
"""

from __future__ import annotations

import json
import ssl
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from typing import Any

from ml_stack.fleet.framing import Limited, LimitedServer, Malformed, read_body

__all__ = ["Call", "Dispatch", "Listener", "Reply", "json_reply"]

SEND_PIECE = 1 << 20


@dataclass(slots=True)
class Call:
    """A request as the route functions see it."""

    method: str
    path: str
    headers: Any
    client: str
    secure: bool
    body: Callable[[int], bytes]
    """Reads the request body, refusing more than the given number of bytes."""

    def json(self, most: int = 16 * 1024) -> dict[str, Any]:
        data = json.loads(self.body(most) or b"{}")
        if not isinstance(data, dict):
            raise Malformed(400, "the body is a JSON object")
        return data


@dataclass(slots=True)
class Reply:
    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    content_type: str = "application/octet-stream"
    stream: Iterator[bytes] | None = None
    length: int = 0
    """The size of ``stream``, which is sent in place of ``body``."""


Dispatch = Callable[[Call], Reply]


def json_reply(status: int, document: dict[str, Any]) -> Reply:
    return Reply(status, json.dumps(document).encode(), content_type="application/json")


def _handler(dispatch: Dispatch) -> type[BaseHTTPRequestHandler]:
    class Handler(Limited, BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: object) -> None:
            return

        def _serve(self) -> None:
            call = Call(self.command, self.path, self.headers, self.client_address[0],
                        isinstance(self.connection, ssl.SSLSocket),
                        lambda most: read_body(self, most))
            try:
                reply = dispatch(call)
            except Malformed as bad:
                self.close_connection = True
                reply = json_reply(bad.status, {"error": bad.message})
            except ValueError:
                reply = json_reply(400, {"error": "the body is not JSON"})
            self.send_response(reply.status)
            self.send_header("Content-Type", reply.content_type)
            self.send_header("Content-Length", str(reply.length if reply.stream else
                                                   len(reply.body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in reply.headers.items():
                self.send_header(key, value)
            self.end_headers()
            if reply.stream is not None:
                for piece in reply.stream:
                    self.wfile.write(piece)
            else:
                self.wfile.write(reply.body)

        do_GET = do_POST = _serve

    return Handler


class Listener:
    """A threaded server running ``dispatch``; TLS when given a context."""

    def __init__(self, dispatch: Dispatch, address: tuple[str, int],
                 context: ssl.SSLContext | None = None) -> None:
        self.httpd = LimitedServer(address, _handler(dispatch), tls=context)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def start(self) -> None:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True,
                                        name="onboard-listener")
        self._thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
