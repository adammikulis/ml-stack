"""Turning what the guard rails report into findings: a denied message is held, a denied tool
call is parked, and a session that keeps being denied is frozen."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from typing import Any

from ml_stack.sentinel.events import Severity
from ml_stack.sentinel.findings import HEURISTIC, HIGH, Finding, finding
from ml_stack.sentinel.rates import Windows

__all__ = ["RailWatch", "reads_like_instruction"]

MARKER = "reads like an instruction"


def reads_like_instruction(reason: str) -> bool:
    """Whether a rail's reason says the text reads like an instruction to the model. A rail
    marking text tainted because its source is external says nothing about the text."""
    return MARKER in reason.lower()


class RailWatch:
    """Counts rail verdicts per session."""

    def __init__(self, *, deny_limit: int = 5, window_s: float = 300.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.deny_limit = deny_limit
        self.windows = Windows(window_s, clock=clock)

    def denied_text(self, session: str, rail: str, reason: str, source: str, text: str,
                    ) -> list[Finding]:
        """A rail refused ``text`` coming from ``source``: hold it, and count the denial."""
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        out = [finding("guard.text_denied", Severity.WARNING, ("message", f"{session}:{digest}"), HIGH,
                       {"rail": rail, "reason": reason, "source": source, "bytes": len(text)},
                       text=text)]
        return out + self._count(session, rail)

    def denied_call(self, session: str, rail: str, reason: str, tool: str,
                    arguments: Mapping[str, Any] | None) -> list[Finding]:
        """A rail refused a tool call: park it with its arguments, and count the denial."""
        body = json.dumps({"tool": tool, "arguments": arguments}, default=str, sort_keys=True)
        digest = hashlib.sha256(body.encode()).hexdigest()[:12]
        out = [finding("guard.call_denied", Severity.WARNING, ("tool_call", f"{session}:{tool}:{digest}"), HIGH,
                       {"rail": rail, "reason": reason, "tool": tool}, text=body)]
        return out + self._count(session, rail)

    def tainted(self, session: str, rail: str, reason: str) -> list[Finding]:
        """A rail marked the session's context as carrying outside instructions. It is
        watched; five of these inside the window freeze the session in ``enforce`` mode."""
        return [finding("guard.tainted", Severity.NOTICE, ("session", session),
                        HEURISTIC, {"rail": rail, "reason": reason}),
                *self._count(session, rail)]

    def tainted_text(self, session: str, rail: str, reason: str, text: str) -> list[Finding]:
        """A rail let ``text`` through marked as carrying outside instructions: the session
        is watched, and the text is held where the mode lets heuristics act."""
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        return [*self.tainted(session, rail, reason),
                finding("guard.tainted_text", Severity.NOTICE, ("message", f"{session}:{digest}"),
                        HEURISTIC, {"rail": rail, "reason": reason, "bytes": len(text)},
                        text=text)]

    def noted(self, session: str, rail: str, action: str, reason: str, source: str,
              ) -> list[Finding]:
        """A rail's logged verdict, without the text: a denial is counted, a modify marks the
        context as carrying outside instructions."""
        if action == "deny":
            return [finding("guard.denied", Severity.WARNING, ("session", session), HEURISTIC, {"rail": rail, "reason": reason, "source": source}),
                    *self._count(session, rail)]
        return self.tainted(session, rail, reason) if reads_like_instruction(reason) else []

    def _count(self, session: str, rail: str) -> list[Finding]:
        n = self.windows.add(session, "denied")
        if n == self.deny_limit:
            return [finding("guard.repeated_denials", Severity.WARNING, ("session", session), HEURISTIC, {"denials": n, "rail": rail})]
        return []
