"""An agent run under sentinel: tool calls are checked before they run, tool results before the
model reads them, and a session sentinel froze runs nothing more."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ml_stack import sentinel
from ml_stack.guard import Unguarded
from ml_stack.interventions import Screened
from ml_stack.sentinel import Sentinel
from ml_stack.sentinel.adapters import GuardLogHandler, agent_gate, rail_answer

__all__ = ["Unwatched", "Watch", "current_session", "resolve", "unwatched"]

CALLER = "agent"

_CURRENT: ContextVar[Watch | None] = ContextVar("ml_stack_agent_watch", default=None)
_TAPS: set[str] = set()


def current_session() -> str:
    """The session of the agent running in this context, or an empty string."""
    current = _CURRENT.get()
    return current.session if current is not None else ""


def _tap(node: Sentinel) -> None:
    """Attach the `GuardLogHandler` of ``node`` to the guard's logger, once. It reads the
    session of the agent whose context is current, so every denial the rails log for a running
    agent counts toward that session's score and a log line from anything else is ignored."""
    where = str(node.root)
    if where in _TAPS:
        return

    def session() -> str:
        current = _CURRENT.get()
        return current.session if current is not None and current.node is node else ""

    _TAPS.add(where)
    node.rails.tapped = True
    logging.getLogger("ml_stack.guard").addHandler(GuardLogHandler(node, session,
                                                                   only_known=True))


@dataclass(frozen=True, slots=True)
class Unwatched:
    """The marker that an agent is meant to run without sentinel."""

    because: str


def unwatched(because: str) -> Unwatched:
    """Run an `Agent` without sentinel. Needs a ``because``; the opt-out is logged."""
    if not because.strip():
        raise ValueError("running an agent without sentinel needs a because=")
    return Unwatched(because.strip())


class Watch:
    """One agent's view of a sentinel, under the session name its findings carry."""

    def __init__(self, node: Sentinel, session: str = "") -> None:
        self.node, self.session = node, session or f"agent-{uuid4().hex[:10]}"
        self._gate = agent_gate(node)
        _tap(node)

    def enter(self) -> None:
        """Make this the agent whose denials the guard's log lines are counted for."""
        _CURRENT.set(self)

    def leave(self) -> None:
        _CURRENT.set(None)

    def running(self) -> contextlib.AbstractContextManager[None]:
        """Marks a tool call of this session as in flight (see `Sentinel.running`)."""
        return self.node.running(self.session)

    def said(self, text: str) -> str:
        """The model's reply as it may be returned and stored: labelled or held when it carries
        a decoy value, held content or a secret-shaped string."""
        return self.node.screen_output(text, session=self.session).text

    def frozen(self) -> str:
        """Why the whole run must stop, or an empty string."""
        return f"session {self.session} is frozen by sentinel" \
            if self.node.session_frozen(self.session) else ""

    def refuses(self, tool: str, arguments: Mapping[str, Any] | None) -> str:
        """Why the call must not run, or an empty string. A decoy it names freezes the session."""
        return self._gate(tool, arguments, session=self.session, caller=CALLER)

    def denied(self, tool: str, arguments: Mapping[str, Any] | None, rail: str, reason: str,
               *, logged: bool = True) -> None:
        """Tell sentinel that a rail refused the call, so it is held and counted. A refusal the
        guard logged as a Deny is counted by the log handler; any other (a Confirm nobody
        answered) is counted here."""
        rails = self.node.rails
        self.node.handle_all(rails.parked_call(self.session, rail, reason, tool, arguments))
        if not logged:
            self.node.handle_all(rails.counted(self.session, rail))
            self.node.note("session", self.session, "guard.denied")

    def shown(self, tool: str, original: str, screened: Screened) -> str:
        """What the model may read of a tool result: the rails' text, or a placeholder when
        sentinel holds the result."""
        got = self.node.screen(original, f"tool:{tool}", session=self.session,
                               verdict=lambda _text, _source: rail_answer(screened))
        return got.text if got.withheld else screened.text


def resolve(interventions: Any, choice: Any) -> Watch | None:
    """The `Watch` an agent runs under: sentinel by default, none when the agent was given
    `unwatched` or ``guard.off`` (the opt-out is logged), none when sentinel's mode is off."""
    because = choice.because if isinstance(choice, Unwatched) \
        else getattr(interventions, "because", "") if isinstance(interventions, Unguarded) else ""
    if because:
        sentinel.opt_out(sentinel.default(), "an agent", because)
        return None
    if isinstance(choice, Watch):
        return choice
    node = choice if isinstance(choice, Sentinel) else sentinel.armed()
    if node.mode == sentinel.Mode.OFF:
        return None
    return Watch(node)
