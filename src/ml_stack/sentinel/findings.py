"""What a detector hands the policy: an event, the subject it concerns, and how sure it is."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ml_stack.sentinel.events import Event, Severity

__all__ = ["HEURISTIC", "HIGH", "Finding", "finding"]

HIGH = "high"
"""A deterministic check that cannot fire on an unmodified install."""

HEURISTIC = "heuristic"
"""A threshold that trades missed attacks against false alarms."""


@dataclass(frozen=True, slots=True)
class Finding:
    """``event`` is what is logged. ``kind`` and ``key`` name the subject a quarantine would
    hold; ``path`` and ``move`` say which file to move aside; ``text`` is content to hold."""

    event: Event
    kind: str
    key: str
    confidence: str
    path: str = ""
    move: str = "file"
    text: str | None = None


def finding(event_kind: str, severity: Severity, subject: tuple[str, str], confidence: str,
            evidence: Mapping[str, Any], **hold: Any) -> Finding:
    """A `Finding` whose event is built from the arguments; ``hold`` may carry ``path``,
    ``move`` and ``text``. The event's source is the part of its kind before the dot."""
    kind, key = subject
    source = event_kind.split(".", 1)[0]
    return Finding(Event(event_kind, severity, source, f"{kind}:{key}", dict(evidence)),
                   kind, key, confidence, **hold)
