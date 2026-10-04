"""The Board's read-only local route: the person's boards, threads and conversations as JSON.

``respond`` answers one request; ``serve`` runs it on a loopback socket with the Board page
and the ml-ui assets. The page holds no token: the person's identity is read here, from the
workspace's own owner token file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ml_stack import activity
from ml_stack.fleet.onboard.web import Call, Listener, Reply as WebReply
from ml_stack.graph.guard import host_ok, refusal
from ml_stack.ui import assets
from ml_stack.workspace import plain, tokens
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.screen import Refused
from ml_stack.workspace.service import Workspace

__all__ = ["MAX_BODY", "PREFIX", "Request", "respond", "serve"]

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
    if method != "GET":
        _, headers, blob = _json(405, {"error": "this route only reads"})
        return 405, {**headers, "Allow": "GET"}, blob
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


@dataclass(frozen=True, slots=True)
class Request:
    """One request: its method, target, lower-case-keyed headers, the port it came in on, and
    whether the host page has checked that the person is the one asking."""

    method: str
    target: str
    headers: Mapping[str, str] = field(default_factory=dict)
    port: int = 0
    signed_in: bool = True


def respond(ws: Workspace, req: Request) -> Reply:
    """The answer to one request."""
    refused = _checked(req.method, req.headers, req.port)
    if refused:
        return refused
    if not req.signed_in:
        return _json(401, {"error": "sign in first"})
    parts = urlsplit(req.target)
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


def _page(req: Request) -> Reply:
    bad = _checked(req.method, req.headers, req.port)
    if bad:
        return bad
    path = urlsplit(req.target).path
    if path in ("/", "/board"):
        return 200, {**SAFE, "Content-Type": "text/html; charset=utf-8",
                     "Content-Security-Policy": "default-src 'none'; script-src 'self' 'unsafe-inline'; "
                                                "style-src 'self' 'unsafe-inline'; connect-src 'self'"}, PAGE.encode()
    name = path.removeprefix("/ui/ml-ui/")
    found = assets().get(name) if path.startswith("/ui/ml-ui/") else None
    if found is None:
        return 404, SAFE, b""
    kind = "text/css" if name.endswith(".css") else "text/javascript" if name.endswith(".js") \
        else "application/json"
    return 200, {**SAFE, "Content-Type": f"{kind}; charset=utf-8"}, found.read_bytes()


class _Route:
    """The dispatch function of a listener: the Board route, the page and the assets."""

    def __init__(self, ws: Workspace) -> None:
        self.ws = ws
        self.port = 0

    def __call__(self, call: Call) -> WebReply:
        req = Request(call.method, call.path, {k.lower(): v for k, v in call.headers.items()},
                      self.port)
        board = urlsplit(call.path).path.startswith(PREFIX)
        status, headers, body = respond(self.ws, req) if board else _page(req)
        kind = headers.pop("Content-Type", "application/octet-stream")
        return WebReply(status, body, headers, kind)


PAGE = """<!doctype html><meta charset="utf-8"><title>Board</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="/ui/ml-ui/ml-ui.css">
<body style="margin:0;background:var(--ml-bg)"><ml-board endpoint="/board" style="height:100vh"></ml-board>
<script type="module" src="/ui/ml-ui/ml-ui.js"></script>"""


def serve(ws: Workspace, port: int = 0) -> Listener:
    """A loopback listener for the Board page and its route; the caller calls ``start`` and
    ``stop``."""
    route = _Route(ws)
    listener = Listener(route, ("127.0.0.1", port))
    route.port = listener.port
    return listener
