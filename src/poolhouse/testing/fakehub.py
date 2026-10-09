"""A Hugging Face endpoint on a loopback socket: listings, searches and file downloads.

``FakeHub`` serves ``repos`` -- ``{"owner/name": {"path/in/repo": bytes}}`` -- with the
routes `poolhouse.hub.remote` and `poolhouse.hub.pull` ask for. Files are handed out by a
second socket (``cdn_url``) after a redirect, as the real Hub does, and the two record
the authorization header each saw. Set ``HF_ENDPOINT`` to ``url`` to point a pull at it.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import urllib.parse
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from typing import Any

from poolhouse.http import Server
from poolhouse.testing.fakes import http_handler

__all__ = ["FakeHub", "fake_hub"]


class FakeHub:
    """The hub and its CDN. ``gated`` repositories answer 401 without the bearer ``token``.

    ``cut_after`` closes the connection after that many body bytes the first time a file is
    read; ``ignore_range`` answers a range request with the whole file; ``corrupt`` names paths whose served bytes differ from the listed sha256;
    ``redirect`` sends file reads through the CDN socket. ``hub_seen`` and ``cdn_seen`` are
    ``(method, path, range, authorization)`` for every request.
    """

    def __init__(self, repos: dict[str, dict[str, bytes]], *, gated: frozenset[str] = frozenset(),
                 redirect: bool = True) -> None:
        self.repos, self.gated, self.redirect = repos, gated, redirect
        self.token = secrets.token_hex(8)
        self.cut_after: int | None = None
        self.ignore_range = False
        self.corrupt: set[str] = set()
        self.downloads: dict[str, int] = {}
        self.hub_seen: list[tuple[str, str, str, str]] = []
        self.cdn_seen: list[tuple[str, str, str, str]] = []
        self._hub = Server(("127.0.0.1", 0), self._handler(cdn=False))
        self._cdn = Server(("127.0.0.1", 0), self._handler(cdn=True))
        self.url = f"http://127.0.0.1:{self._hub.server_address[1]}"
        self.cdn_url = f"http://127.0.0.1:{self._cdn.server_address[1]}"
        self._threads = [threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.02},
                                         daemon=True)
                         for s in (self._hub, self._cdn)]
        for thread in self._threads:
            thread.start()

    def point(self, monkeypatch: Any) -> None:
        """Point ``$HF_ENDPOINT`` at this hub, and name its CDN socket as a host that may be reached
        and may be on this machine, as a person running against a mirror would. The hub itself is
        the endpoint; without this a redirect to the CDN is refused like any other private host."""
        cdn = self.cdn_url.split("//", 1)[1]
        monkeypatch.setenv("HF_ENDPOINT", self.url)
        monkeypatch.setenv("POOLHOUSE_NET_ALLOW_HOSTS", cdn)
        monkeypatch.setenv("POOLHOUSE_FETCH_ALLOW_HOSTS", cdn)

    def close(self) -> None:
        for server in (self._hub, self._cdn):
            server.shutdown()
            server.server_close()

    def listing(self, repo: str) -> list[dict[str, Any]]:
        rows = []
        for path, body in sorted(self.repos[repo].items()):
            digest = hashlib.sha256(body).hexdigest()
            rows.append({"type": "file", "path": path, "size": len(body),
                         "lfs": {"oid": digest, "size": len(body)}})
        return rows

    def search(self, query: str) -> list[dict[str, Any]]:
        return [{"id": name, "downloads": 1000 - n * 100, "likes": n,
                 "gated": name in self.gated}
                for n, name in enumerate(sorted(self.repos)) if query.lower() in name.lower()]

    def _served(self, repo: str, path: str) -> bytes:
        body = self.repos[repo][path]
        return body[::-1] if f"{repo}/{path}" in self.corrupt else body

    def _handler(self, *, cdn: bool) -> type[BaseHTTPRequestHandler]:
        hub = self

        class _Routes:
            """One request's routing; the request's own members are read through to it."""

            def __init__(self, request: BaseHTTPRequestHandler) -> None:
                self.request = request

            def __getattr__(self, name: str) -> Any:
                return getattr(self.request, name)

            def _send(self, status: int, body: bytes = b"", kind: str = "application/json",
                      extra: dict[str, str] | None = None) -> None:
                sent = {"Content-Type": kind, "Content-Length": str(len(body)), **(extra or {})}
                self.send_response(status)
                for key in sent:
                    self.send_header(key, sent[key])
                self.end_headers()
                if self.command != "HEAD" and body:
                    self.wfile.write(body)

            def _allowed(self, repo: str) -> bool:
                if repo not in hub.gated:
                    return True
                return self.headers.get("Authorization") == f"Bearer {hub.token}"

            def _route(self) -> None:
                seen = (self.command, self.path, self.headers.get("Range") or "",
                        self.headers.get("Authorization") or "")
                (hub.cdn_seen if cdn else hub.hub_seen).append(seen)
                parts = urllib.parse.urlsplit(self.path)
                path = urllib.parse.unquote(parts.path)
                bits = path.strip("/").split("/")
                if bits[:2] == ["api", "models"] and len(bits) == 2:
                    query = urllib.parse.parse_qs(parts.query).get("search", [""])[0]
                    self._send(200, json.dumps(hub.search(query)).encode())
                elif bits[:2] == ["api", "models"] and "tree" in bits:
                    repo = "/".join(bits[2:4])
                    if repo not in hub.repos:
                        self._send(404, b"{}")
                    else:
                        self._send(200, json.dumps(hub.listing(repo)).encode())
                elif "resolve" in bits:
                    self._file("/".join(bits[:2]), "/".join(bits[bits.index("resolve") + 2:]))
                elif cdn and bits[0] == "cdn":
                    self._body("/".join(bits[1:3]), "/".join(bits[3:]))
                else:
                    self._send(404, b"{}")

            def _file(self, repo: str, name: str) -> None:
                if repo not in hub.repos or name not in hub.repos[repo]:
                    self._send(404, b"{}")
                elif not self._allowed(repo):
                    self._send(401, b"{}")
                elif hub.redirect and not cdn:
                    where = f"{hub.cdn_url}/cdn/{repo}/{urllib.parse.quote(name)}"
                    self._send(302, b"", extra={"Location": where})
                else:
                    self._body(repo, name)

            def _body(self, repo: str, name: str) -> None:
                whole = hub._served(repo, name)
                start = 0
                asked = "" if hub.ignore_range else self.headers.get("Range") or ""
                if asked.startswith("bytes="):
                    start = int(asked[6:].split("-")[0] or 0)
                if start >= len(whole) and start:
                    self._send(416, b"")
                    return
                body = whole[start:]
                status = 206 if start else 200
                extra = {"Content-Range": f"bytes {start}-{len(whole) - 1}/{len(whole)}"} \
                    if start else {}
                key = f"{repo}/{name}"
                hub.downloads[key] = hub.downloads.get(key, 0) + 1
                if hub.cut_after is not None and self.command == "GET":
                    cut, hub.cut_after = hub.cut_after, None
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body[:cut])
                    self.wfile.flush()
                    self.request.close_connection = True
                    return
                self._send(status, body, "application/octet-stream", extra)

        return http_handler(lambda request: _Routes(request)._route())


@contextmanager
def fake_hub(repos: dict[str, dict[str, bytes]], **options: Any) -> Iterator[FakeHub]:
    """A `FakeHub` for the block, closed at the end."""
    hub = FakeHub(repos, **options)
    try:
        yield hub
    finally:
        hub.close()
