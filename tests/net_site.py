"""A local HTTP site for the download pipeline's tests: routes are registered per test, and
the server honours Range and If-Range so resumption is real."""

from __future__ import annotations

import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler

from ml_stack.http import Server

EICAR = ("X5O!P%@AP[4\\PZX54(P^)7CC)7}$" + "EICAR-STANDARD-ANTIVIRUS-TEST-FILE" + "!$H+H*")


@dataclass
class Route:
    """What a path answers: status, body, headers, and optionally a body that stops early."""

    body: bytes = b""
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    cut_at: int | None = None
    ranges: bool = False
    drip_s: float = 0.0
    endless: bool = False
    seen: list[dict[str, str]] = field(default_factory=list)


class Site:
    """A server on an ephemeral loopback port with a table of routes."""

    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                route = site.routes.get(self.path.split("?")[0])
                if route is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                route.seen.append({k.lower(): v for k, v in self.headers.items()})
                site.answer(self, route)

        self.server = Server(("127.0.0.1", 0), Handler)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def add(self, path: str, body: bytes = b"", **options: object) -> Route:
        headers = options.pop("headers", {})
        route = Route(body=body, headers=dict(headers), **options)  # type: ignore[arg-type]
        self.routes[path] = route
        return route

    def redirect(self, path: str, to: str, status: int = 302) -> Route:
        return self.add(path, b"", status=status, headers={"Location": to})

    def answer(self, h: BaseHTTPRequestHandler, route: Route) -> None:
        body, status = route.body, route.status
        headers = dict(route.headers)
        start = 0
        wanted = h.headers.get("Range", "")
        if route.ranges and wanted.startswith("bytes=") and status == 200:
            tag = headers.get("ETag", "")
            if not h.headers.get("If-Range") or h.headers.get("If-Range") == tag:
                start = int(wanted[6:].split("-")[0])
                status = 206
                headers["Content-Range"] = f"bytes {start}-{len(body) - 1}/{len(body)}"
                body = body[start:]
        h.send_response(status)
        for name, value in headers.items():
            h.send_header(name, value)
        if route.endless:
            h.send_header("Content-Type", "application/octet-stream")
            h.end_headers()
            try:
                while True:
                    h.wfile.write(b"\0" * 65536)
            except OSError:
                return
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        sent = body if route.cut_at is None else body[:route.cut_at]
        try:
            if route.drip_s:
                for byte in range(len(sent)):
                    h.wfile.write(sent[byte:byte + 1])
                    h.wfile.flush()
                    time.sleep(route.drip_s)
            else:
                h.wfile.write(sent)
        except OSError:
            return
        if route.cut_at is not None:
            h.close_connection = True

    def __enter__(self) -> Site:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()


def gguf_bytes(tensors: int = 0, extra: int = 64) -> bytes:
    """A minimal valid GGUF header followed by ``extra`` bytes."""
    return b"GGUF" + struct.pack("<IQQ", 3, tensors, 0) + b"\0" * extra


def safetensors_bytes(tensors: dict[str, tuple[str, list[int], int, int]] | None = None,
                      data: int = 16) -> bytes:
    """A safetensors file: header JSON for ``tensors`` (dtype, shape, begin, end) and zero data."""
    import json

    header = {name: {"dtype": d, "shape": s, "data_offsets": [b, e]}
              for name, (d, s, b, e) in (tensors or {"w": ("F32", [4], 0, 16)}).items()}
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"\0" * data


Maker = Callable[[], Site]
