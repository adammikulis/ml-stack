"""Put ``src`` on the path, and provide a real HTTP server to test against.

No mocking of the transport. Every client test below runs a real ``http.server`` in a
thread and does a real socket round trip, because the bugs these modules exist to prevent
are transport-shaped: a server that answers /health while still loading, one that ignores
a Range header, one that returns 500 on a concurrent request. A mocked ``urlopen`` cannot
reproduce any of them.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


Handler = Callable[[str, str, bytes], tuple[int, bytes]]
"""``(method, path, body) -> (status, response_body)``"""


class _Server:
    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[tuple[str, str, bytes]] = []
        outer = self

        class _H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _dispatch(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length) if length else b""
                outer.requests.append((self.command, self.path, body))
                try:
                    status, payload = outer.handler(self.command, self.path, body)
                except Exception as exc:  # surface handler bugs as 500s, not hangs
                    status, payload = 500, str(exc).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            do_GET = _dispatch
            do_POST = _dispatch

            def log_message(self, *args: object) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self.port = self._httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_Server":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def server() -> Callable[[Handler], _Server]:
    """Start a real HTTP server with a caller-supplied handler."""
    started: list[_Server] = []

    def start(handler: Handler) -> _Server:
        instance = _Server(handler).__enter__()
        started.append(instance)
        return instance

    yield start

    for instance in started:
        instance.__exit__()


def json_reply(payload: object, status: int = 200) -> tuple[int, bytes]:
    return status, json.dumps(payload).encode()
