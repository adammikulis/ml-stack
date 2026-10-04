"""The Board's read-only local route: the person's boards, threads and conversations as JSON.

``respond`` answers one request; ``serve`` runs it on a loopback socket with the Board page
and the ml-ui assets. The page holds no token: the person's identity is read here, from the
workspace's own owner token file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ml_stack import activity
from ml_stack.graph.guard import host_ok, refusal
from ml_stack.http import Server
from ml_stack.ui import assets
from ml_stack.workspace import plain, tokens
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.screen import Refused
from ml_stack.workspace.service import Workspace

__all__ = ["MAX_BODY", "PREFIX", "respond", "serve"]

PREFIX = "/board/"
MAX_BODY = 512 * 1024
SAFE = {"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store",
        "Content-Security-Policy": "default-src 'none'", "Referrer-Policy": "no-referrer"}
Reply = tuple[int, dict[str, str], bytes]


def _json(status: int, body: Any) -> Reply:
    blob = json.dumps(body, ensure_ascii=True, separators=(",", ":")).encode()
    if len(blob) > MAX_BODY:
        status, blob = 502, b'{"error":"the answer is over the size limit; ask for fewer"}'
    return status, {**SAFE, "Content-Type": "application/json; charset=utf-8"}, blob


def _number(query: Mapping[str, list[str]], key: str, default: int, top: int) -> int:
    raw = (query.get(key) or [""])[0]
    return min(max(int(raw), 0), top) if raw.isdigit() and len(raw) < 12 else default


def _board(query: Mapping[str, list[str]]) -> str:
    name = (query.get("board") or [""])[0]
    if not plain.name_ok(name):
        raise ValueError("board is # then lowercase letters, digits, . _ -")
    return name


def _checked(method: str, headers: Mapping[str, str], port: int) -> Reply | None:
    if method not in ("GET", "HEAD"):
        return _json(405, {"error": "this route only reads"})[0], {"Allow": "GET, HEAD"}, b""
    found = refusal(method, dict(headers), port)
    if found:
        return _json(found[0], {"error": found[1]})
    site = headers.get("sec-fetch-site")
    if site is not None and site not in ("same-origin", "none"):
        return _json(403, {"error": "a request from another site"})
    origin = headers.get("origin")
    if origin is not None and not (urlsplit(origin).scheme == "http"
                                   and host_ok(urlsplit(origin).netloc, port)):
        return _json(403, {"error": "a request from another origin"})
    return None


def respond(ws: Workspace, method: str, target: str, headers: Mapping[str, str], port: int,
            *, signed_in: bool = True) -> Reply:
    """The answer to one request. ``headers`` is keyed by lower-case name; ``signed_in`` is
    the host page's own check that the person is the one asking."""
    refused = _checked(method, headers, port)
    if refused:
        return refused
    if not signed_in:
        return _json(401, {"error": "sign in first"})
    parts = urlsplit(target)
    if not parts.path.startswith(PREFIX):
        return _json(404, {"error": "no such route"})
    query, api = parse_qs(parts.query, max_num_fields=8), ws.board
    try:
        token = tokens.read_file(tokens.directory(ws.base) / tokens.OWNER_FILE)
    except (Denied, OSError):
        return _json(503, {"error": "the person's identity is not set up: run `ml-stack-workspace setup`"})
    try:
        return _json(200, _answer(api, token, parts.path[len(PREFIX):], query))
    except Denied as err:
        return _json(403, {"error": plain.line(err, 200)})
    except (ValueError, Refused) as err:
        return _json(400, {"error": plain.line(err, 200)})


def _answer(api: Any, token: str, route: str, query: Mapping[str, list[str]]) -> Any:
    limit = _number(query, "limit", 100, 200)
    if route == "boards":
        return {"boards": api.list(token)}
    if route == "threads":
        found = _board(query)
        activity.record("board.view", subject=found)
        return {"board": found, "threads": api.threads(token, found, limit)}
    if route == "messages":
        return api.ui_read(token, _board(query), _number(query, "after", 0, 1 << 40), limit)
    if route == "thread":
        return api.ui_thread(token, _number(query, "root", 0, 1 << 40))
    if route == "dms":
        return {"conversations": api.dm_list(token)}
    if route == "dm":
        a, b = (query.get("a") or [""])[0], (query.get("b") or [""])[0]
        return {"a": a, "b": b, "messages": api.ui_dm(token, a, b, limit)}
    if route == "head":
        return {"head": api.ws.bus.log.head()[:16]}
    raise ValueError("no such route")


class _Handler(BaseHTTPRequestHandler):
    workspace: Workspace
    page: bytes = b""

    def _send(self, status: int, headers: Mapping[str, str], body: bytes) -> None:
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _handle(self) -> None:
        port = int(self.server.server_address[1])
        lowered = {k.lower(): v for k, v in self.headers.items()}
        path = urlsplit(self.path).path
        if self.command in ("GET", "HEAD") and not path.startswith(PREFIX):
            bad = _checked(self.command, lowered, port)
            if bad:
                return self._send(*bad[:3])
            if path in ("/", "/board"):
                return self._send(200, {**SAFE, "Content-Type": "text/html; charset=utf-8",
                                        "Content-Security-Policy":
                                        "default-src 'none'; script-src 'self' 'unsafe-inline'; "
                                        "style-src 'self' 'unsafe-inline'; connect-src 'self'"},
                                  self.page)
            name = path.removeprefix("/ui/ml-ui/")
            found = assets().get(name) if path.startswith("/ui/ml-ui/") else None
            if found is not None:
                kind = "text/css" if name.endswith(".css") else "text/javascript" \
                    if name.endswith(".js") else "application/json"
                return self._send(200, {**SAFE, "Content-Type": f"{kind}; charset=utf-8"},
                                  found.read_bytes())
            return self._send(404, SAFE, b"")
        self._send(*respond(self.workspace, self.command, self.path, lowered, port))

    do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _handle

    def log_message(self, *args: Any) -> None:
        return


PAGE = """<!doctype html><meta charset="utf-8"><title>Board</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="/ui/ml-ui/ml-ui.css">
<body style="margin:0;background:var(--ml-bg)"><ml-board endpoint="/board" style="height:100vh"></ml-board>
<script type="module" src="/ui/ml-ui/ml-ui.js"></script>"""


def serve(ws: Workspace, port: int = 0) -> Server:
    """A loopback server for the Board page and its route; the caller runs ``serve_forever``."""
    handler = type("BoardHandler", (_Handler,), {"workspace": ws, "page": PAGE.encode()})
    return Server(("127.0.0.1", port), handler)

