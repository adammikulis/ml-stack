"""The Requests inbox over HTTP for the browser: the page, the list, a short-poll feed and the
answer, behind a browser session.

``RequestsApp.dispatch`` is the whole interface (method, path, headers, body in; status, headers,
body out), so a shell with its own server mounts it under ``/requests``. A session is opened by the
launch key (a person's terminal prints it) or by ``open_session`` for a shell that has already
authenticated its browser; its cookie and CSRF token are delivered only inside the page the
server serves. Every POST needs: a loopback ``Host``, ``application/json``, an ``Origin`` equal to
the page's own origin, ``Sec-Fetch-Site: same-origin`` when the browser sends it, the session
cookie, the CSRF token in ``X-Requests-CSRF``, a body under `MAX_BODY`, a request rate under the
limit and the fingerprint of the words that were shown.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ml_stack import person, requests
from ml_stack.graph.guard import refusal
from ml_stack.requests.model import PENDING, STATES, Request
from ml_stack.ui import assets_dir

__all__ = ["ASSETS", "COOKIE", "CSRF_HEADER", "MAX_BODY", "Reply", "RequestsApp", "Session"]

COOKIE = "ml_requests"
CSRF_HEADER = "x-requests-csrf"
MAX_BODY = 4096
MAX_SESSIONS = 8
MOST_BULK = 50
ANSWERS_PER_WINDOW = 20
WINDOW_S = 10.0
BULK_CHOICES = frozenset({"allow-once", "approve", "deny"})
ASSETS = ("base.js", "ml-requests.js", "ml-ui.css")
"""The files the page loads, served from ml-ui's assets and nothing else."""
CSP = ("default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
HEADERS = {"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
           "Cache-Control": "no-store", "X-Frame-Options": "DENY", "Cross-Origin-Resource-Policy": "same-origin"}
PAGE = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Requests</title>'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="ml-requests-csrf" content="{csrf}"><link rel="stylesheet" href="/requests/assets/ml-ui.css">'
        '</head><body><ml-requests></ml-requests><script type="module" src="/requests/assets/ml-requests.js">'
        '</script></body></html>')
ASSET_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


@dataclass(frozen=True, slots=True)
class Reply:
    """What `RequestsApp.dispatch` answers."""

    status: int
    body: bytes
    type: str = "application/json"
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Session:
    """One browser session: its cookie value, its CSRF token and the times of its recent answers."""

    sid: str
    csrf: str
    answers: list[float] = field(default_factory=list)


def _json(status: int, data: Any, headers: Mapping[str, str] | None = None) -> Reply:
    return Reply(status, json.dumps(data, sort_keys=True).encode(), headers=dict(headers or {}))


def _error(status: int, text: str) -> Reply:
    return _json(status, {"error": text})


def view(request: Request) -> dict[str, Any]:
    """A request as the page shows it: its words, choices, state, and the fingerprint an answer must carry."""
    return {"id": request.id, "kind": request.kind, "state": request.state, "subject": request.subject,
            "reason": request.reason, "agent": request.raised_by.agent, "project": request.raised_by.project,
            "created": request.created, "expires": request.expires, "answer": request.answer,
            "answered_by": request.answered_by, "answered_at": request.answered_at,
            "human_only": request.human_only, "destructive": request.destructive,
            "choices": [{"id": c.id, "label": c.label, "effect": c.effect, "approving": c.approving}
                        for c in request.choices],
            "fingerprint": request.fingerprint}


class RequestsApp:
    """The routes under ``/requests`` over one inbox."""

    def __init__(self, inbox: requests.Inbox | None = None, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.inbox, self.clock = inbox, clock
        self.launch_key = secrets.token_urlsafe(24)
        self._key_used = False
        self.sessions: dict[str, Session] = {}

    # -- sessions --------------------------------------------------------------
    def open_session(self) -> Session:
        """A new browser session (the oldest is dropped past `MAX_SESSIONS`)."""
        if len(self.sessions) >= MAX_SESSIONS:
            self.sessions.pop(next(iter(self.sessions)))
        made = Session(secrets.token_urlsafe(24), secrets.token_urlsafe(24))
        self.sessions[made.sid] = made
        return made

    def _session(self, headers: Mapping[str, str]) -> Session | None:
        for part in headers.get("cookie", "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == COOKIE:
                for sid, held in self.sessions.items():
                    if secrets.compare_digest(sid, value):
                        return held
        return None

    def _exchange(self, key: str) -> Session | None:
        if self._key_used or not key or not secrets.compare_digest(key, self.launch_key):
            return None
        self._key_used = True
        return self.open_session()

    def meta(self, session: Session) -> str:
        """The ``<meta>`` tag a shell puts in its own page so the element can answer for ``session``."""
        return f'<meta name="ml-requests-csrf" content="{session.csrf}">'

    # -- routing ---------------------------------------------------------------
    def dispatch(self, method: str, path: str, headers: Mapping[str, str], body: bytes = b"") -> Reply:
        """Route one request; ``headers`` is keyed by lower-case name."""
        parts = urlsplit(path)
        route = parts.path.rstrip("/") or "/"
        query = parse_qs(parts.query)
        found = self._reasons(method, headers)
        if found:
            return found
        if method == "GET" and route.startswith("/requests/assets/"):
            return self._asset(route.rsplit("/", 1)[1])
        if method == "GET" and route == "/requests":
            return self._page(headers, (query.get("k") or [""])[0])
        session = self._session(headers)
        if session is None:
            return _error(401, "open the Requests page from the link ml-stack printed")
        if method == "GET":
            return self._read(route, query)
        if method == "POST" and route in ("/requests/api/answer", "/requests/api/answer-kind"):
            return self._answer(route, session, headers, body)
        return _error(404, "no such route")

    def _reasons(self, method: str, headers: Mapping[str, str]) -> Reply | None:
        port = _port(headers.get("host", ""))
        found = refusal(method, dict(headers), port)
        if found is not None:
            return _error(*found)
        site = headers.get("sec-fetch-site")
        if method == "POST" and site not in (None, "same-origin"):
            return _error(403, "a request from another site")
        if method == "GET" and site not in (None, "same-origin", "none"):
            return _error(403, "a request from another site")
        if method == "POST":
            origin = headers.get("origin", "")
            if origin != f"http://{headers.get('host', '')}":
                return _error(403, "a POST names its own origin")
        return None

    def _asset(self, name: str) -> Reply:
        if name not in ASSETS:
            return _error(404, "no such file")
        data = (assets_dir() / name).read_bytes()
        return Reply(200, data, ASSET_TYPES[Path(name).suffix], HEADERS)

    def _page(self, headers: Mapping[str, str], key: str) -> Reply:
        session = self._session(headers)
        cookie: dict[str, str] = {}
        if session is None:
            session = self._exchange(key)
            if session is None:
                return _error(401, "open the Requests page from the link ml-stack printed")
            cookie = {"Set-Cookie": f"{COOKIE}={session.sid}; HttpOnly; SameSite=Strict; Path=/requests"}
        return Reply(200, PAGE.format(csrf=session.csrf).encode(), "text/html; charset=utf-8", {**HEADERS, **cookie})

    # -- reading ---------------------------------------------------------------
    def _all(self, **filters: Any) -> list[Request]:
        return requests.list_requests(inbox=self.inbox, **filters)

    def _read(self, route: str, query: dict[str, list[str]]) -> Reply:
        one = lambda name: (query.get(name) or [""])[0][:64]  # noqa: E731
        if route == "/requests/api/list":
            state = one("state")
            if state and state not in STATES:
                return _error(400, "no such state")
            kind = one("kind")
            rows = self._all(state=state, agent=one("agent"), project=one("project"), kind=kind, limit=200)
            return _json(200, {"requests": [view(r) for r in rows]}, HEADERS)
        if route.startswith("/requests/api/show/"):
            found = requests.get(route.rsplit("/", 1)[1][:40], inbox=self.inbox)
            return _json(200, view(found), HEADERS) if found else _error(404, "no such request")
        if route == "/requests/api/feed":
            pending = self._all(state=PENDING)
            rev = hashlib.sha256("".join(r.id + r.fingerprint for r in pending).encode()).hexdigest()[:16]
            return _json(200, {"pending": len(pending), "rev": rev}, HEADERS)
        return _error(404, "no such route")

    # -- answering -------------------------------------------------------------
    def _answer(self, route: str, session: Session, headers: Mapping[str, str], body: bytes) -> Reply:
        if not secrets.compare_digest(headers.get(CSRF_HEADER, ""), session.csrf):
            return _error(403, "the page's token is missing or wrong")
        if len(body) > MAX_BODY:
            return _error(413, "a body over the limit")
        now = self.clock()
        session.answers[:] = [t for t in session.answers if now - t < WINDOW_S]
        if len(session.answers) >= ANSWERS_PER_WINDOW:
            return _error(429, "too many answers; wait a few seconds")
        session.answers.append(now)
        try:
            data = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return _error(400, "the body is not JSON")
        if not isinstance(data, dict):
            return _error(400, "the body is one JSON object")
        if route.endswith("answer-kind"):
            return self._bulk(data)
        return self._one(str(data.get("id", ""))[:40], str(data.get("choice", ""))[:40],
                         str(data.get("fingerprint", ""))[:80])

    def _one(self, ident: str, choice: str, fingerprint: str) -> Reply:
        try:
            done = requests.answer(ident, choice, fingerprint, "ui", requests.Context(inbox=self.inbox))
        except requests.Refused as exc:
            codes = {"unknown": 404, "resolved": 409, "changed": 409, "choice": 400}
            return _error(codes.get(exc.why, 400), str(exc))
        except person.HumanRequired as exc:
            return _error(403, str(exc))
        except requests.Unavailable:
            return _error(503, "the request store is unavailable; nothing was answered")
        return _json(200, view(done), HEADERS)

    def _bulk(self, data: dict[str, Any]) -> Reply:
        choice, items = str(data.get("choice", "")), data.get("items")
        if choice not in BULK_CHOICES or not isinstance(items, list) or not 0 < len(items) <= MOST_BULK:
            return _error(400, "answer-kind takes a choice of allow-once, approve or deny and up to "
                               f"{MOST_BULK} items")
        done: list[str] = []
        for item in items:
            ident = str(item.get("id", ""))[:40] if isinstance(item, dict) else ""
            found = requests.get(ident, inbox=self.inbox)
            if found is None or found.human_only or found.destructive or found.choice(choice) is None:
                return _error(400, "that kind of request is answered one at a time")
        for item in items:
            reply = self._one(str(item.get("id", ""))[:40], choice, str(item.get("fingerprint", ""))[:80])
            if reply.status == 200:
                done.append(str(item["id"]))
        return _json(200, {"answered": done, "skipped": len(items) - len(done)}, HEADERS)


def _port(host: str) -> int:
    tail = host.rpartition(":")[2]
    return int(tail) if tail.isdigit() else 80
