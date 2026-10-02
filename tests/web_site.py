"""A real HTTP server on 127.0.0.1 serving a table of routes, for the web-research tests.

A route is ``path -> (status, headers, body)`` or a callable taking the request path and
returning one. ``site.hits`` lists every request path in order, ``site.times`` when each
arrived.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler

from ml_stack.http import Server

Route = tuple[int, dict[str, str], bytes]


class Site:
    def __init__(self) -> None:
        self.routes: dict[str, Route | Callable[[str], Route]] = {}
        self.hits: list[str] = []
        self.times: list[float] = []
        outer = self

        class _H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                outer.hits.append(self.path)
                outer.times.append(time.monotonic())
                route = outer.routes.get(self.path) or outer.routes.get(self.path.split("?")[0])
                if callable(route):
                    route = route(self.path)
                status, headers, body = route or (404, {}, b"not here")
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self._httpd = Server(("127.0.0.1", 0), _H)
        self.base = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self._thread.start()

    def page(self, path: str, html: str) -> None:
        self.routes[path] = (200, {"Content-Type": "text/html; charset=utf-8"}, html.encode())

    def file(self, path: str, body: bytes, content_type: str, **headers: str) -> None:
        self.routes[path] = (200, {"Content-Type": content_type, **headers}, body)

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def serving():
    """A running ``Site``, closed when the generator is."""
    held = Site()
    yield held
    held.close()


def allow_all(url: str) -> str:
    """A URL guard that lets the loopback server through."""
    return url
