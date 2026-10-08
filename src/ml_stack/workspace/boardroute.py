"""The person's Board reads and posts through the local HTTP route.

``respond`` answers one request; ``serve`` runs it on a loopback socket with the Board page
and the ml-ui assets. The page holds no token: the person's identity is read here, from the
workspace's own owner token file, for a browser that opened the page with the listener's session
secret.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ml_stack import activity
from ml_stack.fleet.onboard.web import Call, Listener, Reply as WebReply
from ml_stack.graph.guard import refusal
from ml_stack.ui import assets
from ml_stack.workspace import coordinator_config, plain, tokens
from ml_stack.workspace.identity import HUMAN, Denied
from ml_stack.workspace.rates import RateLimited
from ml_stack.workspace.screen import Refused
from ml_stack.workspace.service import Workspace

__all__ = ["COOKIE", "MAX_BODY", "PREFIX", "Request", "respond", "serve", "session_ok"]

PREFIX = "/board/"
COOKIE = "ml_session"
MAX_BODY = 512 * 1024
POST_MAX = 32 * 1024
WAIT_MAX_S = 25
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


def _checked(method: str, headers: Mapping[str, str], port: int, writes: bool = False) -> Reply | None:
    if method != "GET" and not (writes and method == "POST"):
        _, headers, blob = _json(405, {"error": "this route only reads"})
        return 405, {**headers, "Allow": "GET"}, blob
    if method == "POST" and (headers.get("origin") is None or headers.get("sec-fetch-site", "same-origin") != "same-origin"):
        return _json(403, {"error": "a post comes from this page"})
    found = refusal(method, dict(headers), port, schemes=("http", "https"))
    if found:
        return _json(found[0], {"error": found[1]})
    site = headers.get("sec-fetch-site")
    if site is not None and site not in ("same-origin", "none"):
        return _json(403, {"error": "a request from another site"})
    origin = headers.get("origin")
    if origin is not None and origin not in {
            f"http://{headers.get('host', '')}", f"https://{headers.get('host', '')}"}:
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
    body: bytes = b""


def respond(ws: Workspace, req: Request, *, actor=None) -> Reply:
    """The answer to one request."""
    parts = urlsplit(req.target)
    refused = _checked(req.method, req.headers, req.port, parts.path in (PREFIX + "post", PREFIX + "ack"))
    if refused:
        return refused
    if not req.signed_in:
        return _json(401, {"error": "sign in first"})
    if not parts.path.startswith(PREFIX):
        return _json(404, {"error": "no such route"})
    configured = coordinator_config.load(ws.base)
    if configured.get('mode') == 'remote':
        return _json(409, {'error': 'This device follows a shared coordinator; open its person Board.',
                           'coordinator': configured['endpoint'], 'workspace': configured['workspace']})
    api = ws.board
    try:
        token = actor if actor is not None else tokens.read_file(tokens.directory(ws.base) / tokens.OWNER_FILE)
    except (Denied, OSError):
        return _json(503, {"error": "the person's identity is not set up: run `ml-stack-workspace setup`"})
    try:
        if ws.auth(token).role != HUMAN:
            raise Denied("only the person reads or posts from the page")
        query = parse_qs(parts.query, max_num_fields=8)
        route = parts.path[len(PREFIX):]
        if req.method == "POST":
            return _json(200, _ack(ws, token, req.body) if route == "ack" else _post(ws, token, req.body))
        if route == "file":
            return _file(ws, token, query)
        return _json(200, _answer(api, token, route, query))
    except Denied as err:
        return _json(403, {"error": plain.line(err, 200)})
    except (ValueError, Refused) as err:
        return _json(400, {"error": plain.line(err, 200)})
    except RateLimited as err:
        return _json(429, {"error": plain.line(err, 200)})


def _file(ws: Workspace, token: str, query: Mapping[str, list[str]]) -> Reply:
    """A file for the person: plain text as a download, never rendered; a binary file as its
    name, size and hash only."""
    meta, data = ws.files.content_for_person(token, (query.get("id") or [""])[0])
    if not meta["text"]:
        return _json(200, {"name": plain.line(meta["name"], 80), "size": meta["size"],
                           "type": meta["type"], "sha256": hashlib.sha256(data).hexdigest()})
    name = re.sub(r"[^A-Za-z0-9._-]", "_", str(meta["name"]))[:80] or "file"
    activity.record("board.file", subject=meta["id"], meta={"size": meta["size"]})
    return 200, {**SAFE, "Content-Type": "text/plain; charset=utf-8",
                 "Content-Disposition": f'attachment; filename="{name}.txt"'}, data


def _answer(api: Any, token: str, route: str, query: Mapping[str, list[str]]) -> Any:
    limit = _number(query, "limit", 100, 200)
    if route == "boards":
        return {"me": api.ws.auth(token).id, "boards": api.list(token)}
    if route == "agents":
        me = api.ws.auth(token)
        if me.role != HUMAN:
            raise Denied("only the person reads the agent directory from the page")
        return {"owner_id": me.id, "agents": [
            {"id": row["id"], "role": row["role"], "device": row["device"],
             "display_name": row["display_name"], "session_kind": row["session_kind"],
             "parent": row["parent"], "coordinator_eligible": row["coordinator_eligible"],
             "coordinator_reason": row["coordinator_reason"]}
            for row in api.ws.registered()[:200] if row["id"] != me.id]}
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
        return {"seq": api.ws.news(token, 0, 0)}
    if route == "wait":
        after = _number(query, "after", 0, 1 << 40)
        return {"seq": api.ws.news(token, after, min(_number(query, "timeout", 20, WAIT_MAX_S), WAIT_MAX_S))}
    raise ValueError("no such route")


def _post(ws: Workspace, token: str, raw: bytes) -> dict[str, Any]:
    """The person's one write: a message to a board, thread or conversation."""
    if len(raw) > POST_MAX:
        raise ValueError("the message is too large")
    try:
        doc = json.loads(raw or b"{}")
    except ValueError as err:
        raise ValueError("the body is not JSON") from err
    if not isinstance(doc, dict) or not isinstance(doc.get("to"), str) \
            or not isinstance(doc.get("body"), str) or not doc["body"].strip():
        raise ValueError("a post needs a string `to` and a non-empty string `body`")
    if set(doc) - {"to", "body", "subject", "reply_to", "type"}:
        raise ValueError("unknown post field")
    kind, subject, reply = doc.get("type", "note"), doc.get("subject", ""), doc.get("reply_to", 0)
    if not isinstance(kind, str) or not isinstance(subject, str) or not isinstance(reply, int) \
            or isinstance(reply, bool):
        raise ValueError("type and subject are strings and reply_to is a message number")
    me = ws.auth(token)
    if me.role != HUMAN:
        raise Denied("only the person posts from the page")
    to = doc["to"]
    if to == "#announcements":
        raise Denied("announcements are posted with the workspace announce command")
    if to.startswith("#") and not ws.board.can_post(me, to):
        ws.board.join(token, to)
    sent = ws.send(token, to, kind, doc["body"], subject=subject, reply_to=reply)
    activity.record("board.post", subject=to if to.startswith("#") else "dm", meta={"size": len(doc["body"])})
    return {"seq": sent["seq"], "to": to, "thread": sent["thread"]}


def _ack(ws: Workspace, token: str, raw: bytes) -> dict[str, Any]:
    if len(raw) > POST_MAX:
        raise ValueError("the request is too large")
    doc = json.loads(raw)
    if not isinstance(doc, dict) or set(doc) not in ({"board", "through"}, {"to", "through"}):
        raise ValueError("ack needs through and one board or recipient")
    seq = doc["through"]
    if type(seq) is not int or seq <= 0:
        raise ValueError("through is a positive message number")
    me = ws.auth(token)
    if me.role != HUMAN:
        raise Denied("only the person marks page messages read")
    name = doc.get("board", doc.get("to"))
    if not isinstance(name, str):
        raise ValueError("board or recipient must be text")
    if "board" in doc:
        visible = ws.board.ui_read(token, name, seq - 1, 1)["messages"]
        key = name
    else:
        visible = ws.board.ui_dm(token, me.id, name, 200)
        key = "dm:" + name
    if not any(message["seq"] == seq for message in visible):
        raise Denied("through must identify a visible message in this conversation")
    ws.board.store.mark(me.id, key, seq)
    return doc


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


def session_ok(headers: Mapping[str, str], secret: str) -> bool:
    """Whether the request's cookie holds the page server's session secret."""
    for part in headers.get("cookie", "").split(";"):
        key, _, value = part.strip().partition("=")
        if key == COOKIE and secret and hmac.compare_digest(value.encode(), secret.encode()):
            return True
    return False


class _Route:
    """The dispatch function of a listener: the Board route, the page and the assets. The
    person's browser session is a secret made here; only a browser that opened the page with it
    holds the cookie the Board route requires."""

    def __init__(self, ws: Workspace) -> None:
        self.ws = ws
        self.port = 0
        self.session = secrets.token_urlsafe(24)

    def opened_with_session(self, call: Call) -> bool:
        """Whether the page was requested with this listener's session secret in its query."""
        given = parse_qs(urlsplit(call.path).query).get("session", [""])[0]
        return bool(given) and hmac.compare_digest(given.encode(), self.session.encode())

    def __call__(self, call: Call) -> WebReply:
        headers = {k.lower(): v for k, v in call.headers.items()}
        wanted = int(headers.get("content-length", "0") or 0) if headers.get("content-length", "0").isdigit() else 0
        body = call.body(POST_MAX) if call.method == "POST" and 0 < wanted <= POST_MAX else b""
        req = Request(call.method, call.path, headers, self.port, session_ok(headers, self.session), body)
        board = urlsplit(call.path).path.startswith(PREFIX)
        status, headers, body = respond(self.ws, req) if board else _page(req)
        if status == 200 and not board and self.opened_with_session(call):
            headers["Set-Cookie"] = f"{COOKIE}={self.session}; HttpOnly; SameSite=Strict; Path=/"
        kind = headers.pop("Content-Type", "application/octet-stream")
        return WebReply(status, body, headers, kind)


PAGE = """<!doctype html><meta charset="utf-8"><title>Board</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="/ui/ml-ui/ml-ui.css">
<body style="margin:0;background:var(--ml-bg)"><ml-board endpoint="/board" style="height:100vh"></ml-board>
<script type="module" src="/ui/ml-ui/ml-ui.js"></script>"""


def serve(ws: Workspace, port: int = 0, *, routes: type[_Route] = _Route) -> Listener:
    """A loopback listener for the Board page and its route; the caller calls ``start`` and
    ``stop``, and ``listener.session`` is the secret a browser opens the page with
    (``/?session=...``) to hold the cookie the route requires."""
    route = routes(ws)
    listener = Listener(route, ("127.0.0.1", port))
    route.port = listener.port
    listener.session = route.session  # type: ignore[attr-defined]
    return listener
