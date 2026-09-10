"""The routes beside the ask ones, each a mixin a handler adds when it has what the route
needs: refreshing the sources, the review queue, a change request, and a draft note.

Each is a 404 until the subclass gives it its collaborator, and the two that change what is
on this machine -- review and refresh -- refuse a request that came through a proxy.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ml_stack.graph.payloads import sse
from ml_stack.log import warn

__all__ = ["DraftRoutes", "LocalOnly", "PROXY_HEADERS", "REQUEST_MOST", "RefreshRoutes",
           "RequestRoutes", "ReviewRoutes"]

PROXY_HEADERS = ("X-Forwarded-For", "Forwarded", "Cf-Ray", "CF-Connecting-IP",
                 "Cf-Access-Jwt-Assertion")
"""A request wearing any of these came through a proxy or a tunnel, not from this machine.
The routes that change what is on this machine -- review, refresh -- refuse it."""

# the most a change request may carry, by field
REQUEST_MOST = {"kind": 60, "claimed": 120, "claimedLabel": 120, "text": 2000}


class LocalOnly:
    """`proxied`, for a route that answers only a request made from this machine."""

    def proxied(self) -> bool:
        """Whether the request wears a proxy or tunnel header (`PROXY_HEADERS`)."""
        headers = getattr(self, "headers", None)
        return headers is not None and any(headers.get(h) is not None for h in PROXY_HEADERS)

    def refused(self) -> bool:
        """A 403 for a proxied request; True when one was sent."""
        if not self.proxied():
            return False
        self.send_error(403)
        return True


class RefreshRoutes(LocalOnly):
    """``GET /refresh``: a re-read of the sources, streamed stage by stage.

    A subclass gives ``stages()``, a generator of ``(stage, detail)`` pairs that ends with
    ``done`` or ``error``; the page shows each as it arrives and reloads on ``done``. One
    runs at a time: a second request while one runs is a 409. Without ``stages`` the route
    is a 404, and a proxied request is refused.
    """

    def stages(self) -> Iterator[tuple[str, str]] | None:
        return None

    @classmethod
    def _refresh_lock(cls) -> threading.Lock:
        lock = cls.__dict__.get("_refreshing")
        if lock is None:
            lock = threading.Lock()
            setattr(cls, "_refreshing", lock)
        return lock

    def handle_refresh(self) -> None:
        if self.refused():
            return
        stages = self.stages()
        if stages is None:
            self.send_error(404)
            return
        lock = type(self)._refresh_lock()
        if not lock.acquire(blocking=False):
            self.send_error(409)
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            for stage, detail in stages:
                sse(self.wfile, {"stage": str(stage), "detail": str(detail), "t": time.time()})
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            lock.release()


class ReviewRoutes(LocalOnly):
    """``GET /review`` and ``POST /review``: the proposals waiting for a person.

    ``queue`` is a `ml_stack.graph.review.Queue`; the list is ``queue.listed()`` and a POST
    of ``{"id", "action"}`` is ``queue.act``: 404 for an id it does not hold, 400 for an
    action it does not offer, else ``{"ok": true, "problems": [...]}``. Without a queue both
    are 404, and a proxied request is refused.
    """

    queue: Any = None

    def handle_review_list(self) -> None:
        if self.refused():
            return
        if self.queue is None:
            self.send_error(404)
            return
        self.send_json(200, self.queue.listed())

    def handle_review_act(self, body: Mapping[str, Any]) -> None:
        if self.refused():
            return
        if self.queue is None:
            self.send_error(404)
            return
        from ml_stack.graph.review import ACTIONS

        action = str(body.get("action") or "")
        if action not in ACTIONS:
            self.send_json(400, {"error": "action must be " + ", ".join(ACTIONS)})
            return
        try:
            problems = self.queue.act(str(body.get("id") or ""), action)
        except KeyError:
            self.send_json(404, {"error": "no such proposal"})
            return
        self.send_json(200, {"ok": True, "problems": problems})


class RequestRoutes:
    """``POST /request``: a change request from the page's form, kept before it is read.

    ``requests`` is the JSONL file rows are appended to; the row is checked and cut to
    `REQUEST_MOST` first, a request with no words is a 400, and 204 goes back as soon as
    the row is on disk. Then ``proposed(row)`` runs, after the response, for a subclass
    that asks a model what the request means (`ml_stack.graph.requests.propose`).
    Without ``requests`` the route is a 404.
    """

    requests: Path | None = None

    def proposed(self, row: Mapping[str, Any]) -> None:
        return None

    def handle_request(self, body: Mapping[str, Any] | None) -> None:
        if self.requests is None:
            self.send_error(404)
            return
        if body is None or not str(body.get("text") or "").strip():
            self.send_error(400)
            return
        row = {"at": str(body.get("at") or ""),
               "kind": str(body.get("kind") or "")[:REQUEST_MOST["kind"]],
               "claimed": str(body.get("claimed") or "")[:REQUEST_MOST["claimed"]],
               "claimedLabel": str(body.get("claimedLabel") or "")[:REQUEST_MOST["claimedLabel"]],
               "attested": bool(body.get("attested")),
               "text": str(body["text"])[:REQUEST_MOST["text"]],
               "targets": list(body.get("targets") or [])}
        self.requests.parent.mkdir(parents=True, exist_ok=True)
        with self.requests.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.send_response(204)
        self.end_headers()
        try:
            self.wfile.flush()
        except OSError:
            pass
        self.proposed(row)


class DraftRoutes:
    """``POST /draft``: a note introducing the entries an answer named, for a person to send.

    ``drafter(ids, question, answer)`` is the subclass's: `ml_stack.graph.conversation.draft`
    on the graph and a slot of the model, returning its dict. 400 without ids, 404 without a
    drafter, 500 with the exception's text when it raised.
    """

    def drafter(self, ids: list[str], question: str, answer: str) -> Mapping[str, Any] | None:
        return None

    def handle_draft(self, body: Mapping[str, Any]) -> None:
        ids = body.get("ids")
        if not (isinstance(ids, list) and all(isinstance(i, str) for i in ids)):
            self.send_json(400, {"error": "no ids"})
            return
        try:
            out = self.drafter(list(ids), str(body.get("question") or ""),
                               str(body.get("answer") or ""))
        except Exception as exc:  # noqa: BLE001 - the page shows the reason
            warn(f"{time.strftime('%FT%T')} draft failed: {exc}")
            self.send_json(500, {"error": str(exc)[:200]})
            return
        if out is None:
            self.send_error(404)
            return
        self.send_json(200, dict(out))
