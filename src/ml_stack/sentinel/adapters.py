"""Hooks that feed other modules' events to a sentinel without importing them.

Each takes the other module's object and wraps or listens to it: the guard's logger, a MAC
`Authenticator`, the Broker's event callback, an agent loop's tool-call check.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from ml_stack.sentinel.core import Sentinel
from ml_stack.sentinel.events import Event, Severity
from ml_stack.sentinel.human import agent_may

__all__ = ["GuardLogHandler", "agent_gate", "broker_listener", "note_refusal",
           "watch_authenticator"]

_OUTCOMES = (("already seen", "replay"), ("outside the window", "clock"),
             ("too many failures", "locked"), ("not signed", "bad_sig"))


class GuardLogHandler(logging.Handler):
    """Reads ``ml_stack.guard`` warnings (``rail, action, reason, source``) as findings for
    ``session``. Attach it with ``logging.getLogger("ml_stack.guard").addHandler``."""

    def __init__(self, sentinel: Sentinel, session: Callable[[], str] = lambda: "") -> None:
        super().__init__(logging.WARNING)
        self.sentinel, self.session = sentinel, session

    def emit(self, record: logging.LogRecord) -> None:
        args = record.args
        if not (isinstance(args, tuple) and len(args) == 4):
            return
        rail, action, reason, source = (str(a) for a in args)
        self.sentinel.handle_all(self.sentinel.rails.noted(
            self.session() or "none", rail, action, reason, source))


def watch_authenticator(auth: Any, sentinel: Sentinel, verdict: type) -> Any:
    """Wrap ``auth.check`` (a ``macauth.Authenticator``; ``verdict`` is its ``Verdict``
    class) so that outcomes feed the peer watch and a quarantined address is refused."""
    original = auth.check

    def check(method: str, url: str, headers: Mapping[str, str], body: bytes | None,
              who: str = "") -> Any:
        if who and sentinel.peer_blocked(who):
            return verdict(False, "this address is quarantined", locked=True)
        result = original(method, url, headers, body, who)
        if who:
            outcome = "ok" if result.ok else next(
                (o for needle, o in _OUTCOMES if needle in result.reason), "bad_sig")
            sentinel.handle_all([sentinel.peers.note(who, outcome)])
        return result

    auth.check = check
    return auth


def note_refusal(sentinel: Sentinel, host: str, reason: str) -> None:
    """Record that ``httpguard`` refused to fetch from ``host``."""
    sentinel.bus.emit(Event("httpguard.refused", Severity.NOTICE, "httpguard",
                            f"host:{host}", {"reason": reason}, sentinel.clock()))


def broker_listener(sentinel: Sentinel) -> Callable[[str, dict[str, object]], None]:
    """A callback with the Broker's ``on_event(event, fields)`` shape. Every event becomes
    an info event; a ``caller`` field also counts toward that caller's resource use."""
    def on(event: str, fields: dict[str, object]) -> None:
        caller = str(fields.get("caller") or "")
        sentinel.bus.emit(Event(f"broker.{event}", Severity.INFO, "broker",
                                f"caller:{caller}" if caller else "",
                                {k: v for k, v in fields.items() if k != "caller"},
                                sentinel.clock()))
        if caller:
            sentinel.handle_all([sentinel.abuse.note(caller)])
    return on


def agent_gate(sentinel: Sentinel) -> Callable[..., str]:
    """A check for an agent loop to call before running a tool: why the call must not
    run, or an empty string."""
    def gate(tool: str, arguments: Mapping[str, Any] | None = None, *, session: str = "",
             caller: str = "") -> str:
        return agent_may(tool, arguments) or sentinel.screen_call(
            tool, arguments, session=session, caller=caller)
    return gate
