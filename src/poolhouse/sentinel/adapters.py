"""Hooks that feed other modules' events to a sentinel without importing them.

Each takes the other module's object and wraps or listens to it: the guard's logger, a MAC
`Authenticator`, the Broker's event callback, an agent loop's tool-call check.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poolhouse.interventions import Call
from poolhouse.sentinel.core import Sentinel
from poolhouse.sentinel.events import Event, Severity
from poolhouse.sentinel.human import agent_may
from poolhouse.sentinel.policy import Mode
from poolhouse.sentinel.store import Record

__all__ = ["GuardLogHandler", "RailAnswer", "agent_gate", "broker_listener", "note_refusal",
           "rail_answer", "sandbox_listener", "screening", "serve_hooks", "watch_authenticator"]

_OUTCOMES = (("already seen", "replay"), ("outside the window", "clock"),
             ("too many failures", "locked"), ("not signed", "bad_sig"))


class GuardLogHandler(logging.Handler):
    """Reads the ``poolhouse.guard`` warnings a `Run` writes (``rail, verdict, hook``) as
    findings for ``session``: every Deny is a denial counted against the session. Attach it
    with ``logging.getLogger("poolhouse.guard").addHandler``."""

    def __init__(self, sentinel: Sentinel, session: Callable[[], str] = lambda: "", *,
                 only_known: bool = False) -> None:
        super().__init__(logging.WARNING)
        self.sentinel, self.session, self.only_known = sentinel, session, only_known

    def emit(self, record: logging.LogRecord) -> None:
        args = record.args
        if not (isinstance(args, tuple) and len(args) == 3):
            return
        rail, verdict, hook = (str(a) for a in args)
        who = self.session()
        if verdict != "Deny" or (self.only_known and not who):
            return
        self.sentinel.handle_all(self.sentinel.rails.noted(who or "none", rail, "deny", hook,
                                                           hook))
        if who:
            self.sentinel.note("session", who, "guard.denied")


@dataclass(frozen=True, slots=True)
class RailAnswer:
    """What a `Run` said about a text: whether it withheld it, whether it marked the run as
    carrying outside text, and the rail and reason behind it."""

    denied: bool
    tainted: bool
    rail: str
    reason: str


def screening(run: Any, tool: str = "web_fetch") -> Callable[[str, str], RailAnswer]:
    """The ``verdict`` for `Sentinel.screen`: it shows ``run`` (an ``interventions.Run``) a
    tool result from ``source`` and reports what the run's interventions did with it."""
    def verdict(text: str, source: str) -> RailAnswer:
        shown = run.screen_result(Call(tool if source.startswith("tool:") else source), text)
        by = shown.verdict
        return RailAnswer(bool(shown.withheld), bool(getattr(by, "tainted", False)),
                          str(getattr(by, "by", "")), str(getattr(by, "reason", "")))
    return verdict


def rail_answer(shown: Any) -> RailAnswer:
    """The `RailAnswer` for an ``interventions.Screened``: what the rails did with a text."""
    by = shown.verdict
    return RailAnswer(bool(shown.withheld), bool(getattr(by, "tainted", False)),
                      str(getattr(by, "by", "")), str(getattr(by, "reason", "")))


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
            seen = sentinel.abuse.note(caller)
            sentinel.handle_all([seen])
            if seen is not None:
                sentinel.note("caller", caller, "abuse.resource")
            if event == "refused":
                sentinel.note("caller", caller, "broker.refused")
    return on


def sandbox_listener(sentinel: Sentinel) -> Callable[[str, dict[str, object]], None]:
    """A callback with the sandbox's ``on_event(event, fields)`` shape: a refusal, a timeout, a
    missing sandbox or a run without one becomes an event on the bus, and counts toward the score
    of each session that has a tool call running (the only attribution a sandbox run has)."""
    levels = {"warning": Severity.WARNING, "notice": Severity.NOTICE}

    def on(event: str, fields: dict[str, object]) -> None:
        if sentinel.mode == Mode.OFF:
            return
        level = levels.get(str(fields.get("severity", "")), Severity.INFO)
        sentinel.bus.emit(Event(event, level, "sandbox", f"policy:{fields.get('policy', '')}",
                                {k: v for k, v in fields.items() if k != "severity"},
                                sentinel.clock()))
        for session in sentinel.in_flight():
            sentinel.note("session", session, event)
    return on


def serve_hooks(sentinel: Sentinel, servers: Callable[[], Mapping[int, Mapping[str, Any]]],
                stop: Callable[[int], object]) -> None:
    """Quarantining a ``server`` (key ``port:N``) or a ``model`` stops the recorded servers
    that match, through ``stop(port)``. ``servers`` is the lease file's record of servers
    poolhouse started (``serve.recorded_servers``); a process not in it is never touched."""
    def stop_server(record: Record) -> None:
        port = int(record.key.removeprefix("port:"))
        if port in servers():
            stop(port)

    def stop_serving(record: Record) -> None:
        wanted = Path(record.key).resolve()
        for port, entry in dict(servers()).items():
            if entry.get("model") and Path(str(entry["model"])).resolve() == wanted:
                stop(port)

    sentinel.store.on_quarantine.setdefault("server", []).append(stop_server)
    sentinel.store.on_quarantine.setdefault("model", []).append(stop_serving)


def agent_gate(sentinel: Sentinel) -> Callable[..., str]:
    """A check for an agent loop to call before running a tool: why the call must not
    run, or an empty string."""
    def gate(tool: str, arguments: Mapping[str, Any] | None = None, *, session: str = "",
             caller: str = "") -> str:
        return agent_may(tool, arguments) or sentinel.screen_call(
            tool, arguments, session=session, caller=caller)
    return gate
