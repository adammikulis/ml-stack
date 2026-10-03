"""An agent run under sentinel: tool calls are checked before they run, tool results before the
model reads them, and a session sentinel froze runs nothing more."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ml_stack import sentinel
from ml_stack.guard import Unguarded
from ml_stack.interventions import Screened
from ml_stack.sentinel import Sentinel
from ml_stack.sentinel.adapters import agent_gate, rail_answer

__all__ = ["Unwatched", "Watch", "resolve", "unwatched"]

CALLER = "agent"


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

    def __init__(self, node: Sentinel, session: str) -> None:
        self.node, self.session = node, session
        self._gate = agent_gate(node)

    def frozen(self) -> str:
        """Why the whole run must stop, or an empty string."""
        return f"session {self.session} is frozen by sentinel" \
            if self.node.session_frozen(self.session) else ""

    def refuses(self, tool: str, arguments: Mapping[str, Any] | None) -> str:
        """Why the call must not run, or an empty string. A decoy it names freezes the session."""
        return self._gate(tool, arguments, session=self.session, caller=CALLER)

    def denied(self, tool: str, arguments: Mapping[str, Any] | None, rail: str, reason: str,
               ) -> None:
        """Tell sentinel that a rail refused the call, so it is held and counted."""
        self.node.handle_all(self.node.rails.denied_call(self.session, rail, reason, tool,
                                                         arguments))

    def shown(self, tool: str, original: str, screened: Screened) -> str:
        """What the model may read of a tool result: the rails' text, or a placeholder when
        sentinel holds the result."""
        got = self.node.screen(original, f"tool:{tool}", session=self.session,
                               verdict=lambda _text, _source: rail_answer(screened))
        return got.text if got.withheld else screened.text


def resolve(interventions: Any, choice: Any, session: str) -> Watch | None:
    """The `Watch` an agent runs under: sentinel by default, none when the agent was given
    `unwatched` or ``guard.off`` (the opt-out is logged), none when sentinel's mode is off."""
    because = choice.because if isinstance(choice, Unwatched) \
        else getattr(interventions, "because", "") if isinstance(interventions, Unguarded) else ""
    if because:
        sentinel.opt_out(sentinel.default(), "an agent", because)
        return None
    node = choice if isinstance(choice, Sentinel) else sentinel.armed()
    if node.mode == sentinel.Mode.OFF:
        return None
    return Watch(node, session or f"agent-{uuid4().hex[:10]}")
