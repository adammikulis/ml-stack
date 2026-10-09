"""One score per subject: repeated small signals add up, and the sum escalates watch, then quarantine.

A guard rail's denial, a sandbox refusal, a refused lease, a resource-limit hit are each weak
evidence alone. Counted over a window against the session or caller that caused them they
are strong: ``WATCH_AT`` puts the subject on watch, ``QUARANTINE_AT`` holds it (a frozen session,
a blocked caller) until a person releases it. A release starts the count again.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from poolhouse.sentinel.events import Severity
from poolhouse.sentinel.findings import HEURISTIC, HIGH, Finding, finding
from poolhouse.sentinel.rates import Windows

__all__ = ["QUARANTINE_AT", "WATCH_AT", "WEIGHTS", "WINDOW_S", "Score"]

WATCH_AT = 3.0
"""Score at which a subject is watched."""

QUARANTINE_AT = 10.0
"""Score at which a subject is held, in ``guarded`` and ``enforce`` mode."""

WINDOW_S = 600.0
"""Seconds a signal counts for."""

WEIGHTS: dict[str, float] = {
    "guard.denied": 1.0,
    "sandbox.denied": 2.0,
    "sandbox.unavailable": 1.0,
    "sandbox.unsandboxed": 1.0,
    "sandbox.timeout": 0.5,
    "broker.refused": 2.0,
    "abuse.resource": 3.0,
    "server.unmanaged": 0.5,
}
"""What one signal of each kind adds."""


@dataclass(slots=True)
class _Level:
    watching: bool = False
    held: bool = False


class Score:
    """Weighted signals per ``(kind, key)`` inside a sliding window."""

    def __init__(self, *, watch_at: float = WATCH_AT, quarantine_at: float = QUARANTINE_AT,
                 window_s: float = WINDOW_S, clock: Callable[[], float] = time.monotonic) -> None:
        self.watch_at, self.quarantine_at = watch_at, quarantine_at
        self.windows = Windows(window_s, clock=clock)
        self._levels: dict[tuple[str, str], _Level] = {}

    def value(self, kind: str, key: str) -> float:
        """The subject's score now."""
        return self.windows.count(f"{kind}:{key}", "score") / 100

    def held(self, kind: str, key: str) -> bool:
        """Whether this score has already held the subject."""
        level = self._levels.get((kind, key))
        return bool(level and level.held)

    def add(self, kind: str, key: str, signal: str, weight: float | None = None,
            ) -> list[Finding]:
        """Count one ``signal`` against ``(kind, key)``; a finding when a threshold is crossed
        for the first time since the subject was last reset."""
        if weight is None:
            weight = WEIGHTS.get(signal, 1.0)
        total = self.windows.add(f"{kind}:{key}", "score", round(weight * 100)) / 100
        level = self._levels.setdefault((kind, key), _Level())
        evidence = {"score": round(total, 2), "signal": signal,
                    "watch_at": self.watch_at, "quarantine_at": self.quarantine_at}
        if total >= self.quarantine_at and not level.held:
            level.held = level.watching = True
            return [finding("score.quarantine", Severity.CRITICAL, (kind, key), HIGH, evidence)]
        if total >= self.watch_at and not level.watching:
            level.watching = True
            return [finding("score.watch", Severity.WARNING, (kind, key), HEURISTIC, evidence)]
        return []

    def reset(self, kind: str, key: str) -> None:
        """Forget the subject's signals, after a person released it."""
        self._levels.pop((kind, key), None)
        self.windows.forget(f"{kind}:{key}", "score")
