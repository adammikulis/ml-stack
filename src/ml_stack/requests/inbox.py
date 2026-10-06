"""Raising, answering and listing requests: the module every component that waits for a person calls."""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ml_stack import keystore, person
from ml_stack.requests.model import PENDING, Request
from ml_stack.requests.store import Ask, Inbox, Refused, Unavailable, build

__all__ = [
    "Context",
    "Handle",
    "Heard",
    "Outcome",
    "answer",
    "default",
    "get",
    "list_requests",
    "oldest_pending",
    "pending_count",
    "raise_request",
    "subscribe",
    "summary",
]

logger = logging.getLogger("ml_stack.requests")
POLL_S = 0.1
POLL_MOST_S = 0.5
_INBOXES: dict[tuple[str, int], Inbox] = {}
_LOCK = threading.Lock()


def default() -> Inbox:
    """This process's inbox for the state root and keyring in force now."""
    inbox = Inbox()
    key = (str(inbox.directory), id(keystore.default()))
    with _LOCK:
        return _INBOXES.setdefault(key, inbox)


@dataclass(frozen=True, slots=True)
class Outcome:
    """How a request ended: ``state`` (``pending`` only when a wait ran out), the ``choice`` made
    and ``via`` which way. ``approved`` is true only for an approving answer."""

    state: str
    choice: str = ""
    via: str = ""
    why: str = ""

    @property
    def approved(self) -> bool:
        return self.state == "approved"


class Handle:
    """What `raise_request` returns: ``wait`` blocks for the answer, ``withdraw`` cancels it. A
    handle made when the store was unavailable answers ``denied`` at once."""

    def __init__(self, inbox: Inbox | None, request: Request | None, secret: str, why: str = "") -> None:
        self._inbox, self.request, self._secret, self._why = inbox, request, secret, why
        self.id = request.id if request else ""
        self.fingerprint = request.fingerprint if request else ""

    def outcome(self) -> Outcome:
        """The state now, without waiting."""
        if self._inbox is None or self.request is None:
            return Outcome("denied", why=self._why or "the request store is unavailable")
        try:
            got = self._inbox.get(self.id)
        except Unavailable as exc:
            return Outcome("denied", why=str(exc))
        if got is None:
            return Outcome("denied", why="the request is gone")
        return Outcome(got.state, got.answer, got.answered_by)

    def wait(self, timeout: float | None = None, *, withdraw: bool = True,
             stop: Callable[[], bool] | None = None) -> Outcome:
        """The answer, once there is one; an expired request is ``expired`` and so not approved.
        When ``timeout`` runs out first the request is withdrawn (unless ``withdraw`` is false)
        and the outcome says ``pending``."""
        began, pause = time.monotonic(), POLL_S
        while True:
            got = self.outcome()
            if got.state != PENDING:
                return got
            if (timeout is not None and time.monotonic() - began >= timeout) or (stop and stop()):
                if withdraw:
                    self.withdraw()
                    return self.outcome()
                return got
            time.sleep(pause)
            pause = min(POLL_MOST_S, pause * 1.5)

    def withdraw(self) -> bool:
        """Cancel the request; only the holder of this handle can."""
        if self._inbox is None or self.request is None:
            return False
        try:
            done = self._inbox.withdraw(self.id, self._secret)
        except (Unavailable, Refused):
            return False
        if done:
            _record("request.cancelled", self.request, "cancelled", "")
        return done


def raise_request(ask: Ask, *, inbox: Inbox | None = None) -> Handle:
    """Put a question to the person and return the handle to wait on. Never raises: a store
    that cannot be used gives a handle that answers ``denied``. ``ValueError`` only for a kind
    or choice that does not exist."""
    held = inbox or default()
    secret = secrets.token_urlsafe(16)
    request = build(ask, held._now())
    try:
        held.add(request, secret)
    except (Unavailable, Refused) as exc:
        logger.warning("request not stored: %s", exc)
        return Handle(None, request, secret, str(exc))
    _record("request.raised", request, "pending", "")
    return Handle(held, request, secret)


@dataclass(frozen=True, slots=True)
class Context:
    """What an answer is made in: the terminal's tty state and the environment checked for agent
    markers (the process's when None), and the inbox (the user's when None)."""

    terminal: tuple[bool, bool] | None = None
    env: Mapping[str, str] | None = None
    inbox: Inbox | None = None


def answer(ident: str, choice: str, fingerprint: str, via: str, ctx: Context | None = None) -> Request:
    """A person's answer. Refused (`person.HumanRequired`) from a process an agent started, and
    from a ``terminal`` answer that is not at a tty; `Refused` when the request was already
    resolved, changed since ``fingerprint`` was shown or has no such choice;
    `Unavailable` when the store cannot be used (nothing is approved)."""
    ctx = ctx or Context()
    if via == "terminal":
        person.require_person("answering a request", ctx.terminal, ctx.env)
    elif person.marked(ctx.env):
        raise person.HumanRequired(f"answering a request is for a person; this process was started "
                                   f"by an agent ({person.marked(ctx.env)})")
    done = (ctx.inbox or default()).answer(ident, choice, fingerprint, via)
    _record("request.answered", done, done.state, via, actor="person")
    return done


def get(ident: str, *, inbox: Inbox | None = None) -> Request | None:
    """The request called ``ident``; None when unknown or the store cannot be read."""
    try:
        return (inbox or default()).get(ident)
    except Unavailable:
        return None


def list_requests(*, inbox: Inbox | None = None, **filters: Any) -> list[Request]:
    """Requests, pending first, filtered by ``state``, ``agent``, ``project``, ``kind``; empty when
    the store cannot be read."""
    try:
        return (inbox or default()).list(**filters)
    except Unavailable:
        return []


def pending_count(*, inbox: Inbox | None = None) -> int:
    """How many requests wait for a person; 0 when the store cannot be read."""
    return len(list_requests(state=PENDING, inbox=inbox))


def oldest_pending(*, inbox: Inbox | None = None) -> Request | None:
    """The request that has waited longest, or None."""
    found = list_requests(state=PENDING, inbox=inbox)
    return found[0] if found else None


def summary(*, inbox: Inbox | None = None) -> dict[str, Any]:
    """The count by kind of what waits, for a status line or a chip."""
    found = list_requests(state=PENDING, inbox=inbox)
    kinds: dict[str, int] = {}
    for item in found:
        kinds[item.kind] = kinds.get(item.kind, 0) + 1
    return {"pending": len(found), "kinds": kinds,
            "human_only": sum(1 for r in found if r.human_only)}


Heard = Callable[[str, Request, str, str, str], None]
"""``(kind, request, outcome, via, actor)``: what a listener is told of a raise, an answer or a withdrawal."""
_LISTENERS: list[Heard] = []


def subscribe(listener: Heard) -> None:
    """Tell ``listener`` about every request raised, answered or withdrawn in this process from now on."""
    with _LOCK:
        if listener not in _LISTENERS:
            _LISTENERS.append(listener)


def _record(kind: str, request: Request, outcome: str, via: str, *, actor: str = "") -> None:
    with _LOCK:
        heard = list(_LISTENERS)
    for listener in heard:
        try:
            listener(kind, request, outcome, via, actor)
        except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
            logger.debug("request listener failed: %s", type(exc).__name__)
