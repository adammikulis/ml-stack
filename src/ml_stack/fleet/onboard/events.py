"""What onboarding reports about itself: one record per step, for a person and for sentinel.

Every step an attacker's presence would show in (a request, an accept, a decline, an expiry, a
wrong code, a lock, a replay, a chunk that fails its hash) is an `Event` handed to every
subscriber and logged. Sentinel subscribes with a small adapter on its side (docs/onboarding.md
shows it); nothing here imports sentinel.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BUS", "SEVERITIES", "Bus", "Event"]

logger = logging.getLogger("ml_stack.fleet.onboard")
logger.addHandler(logging.NullHandler())

SEVERITIES = ("info", "notice", "warning", "critical")


@dataclass(frozen=True, slots=True)
class Event:
    """``kind`` is ``onboard.<what>``; ``subject`` is ``device:<fingerprint>`` or
    ``request:<id>``; ``evidence`` holds short facts, never a code, key or secret."""

    kind: str
    severity: str
    subject: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class Bus:
    """Hands events to subscribers and keeps the last few for a command to show."""

    def __init__(self, keep: int = 512) -> None:
        self._subscribers: list[Callable[[Event], None]] = []
        self._recent: deque[Event] = deque(maxlen=keep)
        self._lock = threading.Lock()

    def subscribe(self, fn: Callable[[Event], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(fn)

        def stop() -> None:
            with self._lock:
                if fn in self._subscribers:
                    self._subscribers.remove(fn)
        return stop

    def emit(self, kind: str, severity: str, subject: str = "", **evidence: Any) -> Event:
        if severity not in SEVERITIES:
            raise ValueError(f"severity is one of {SEVERITIES}, not {severity!r}")
        event = Event(kind, severity, subject, dict(evidence))
        with self._lock:
            self._recent.append(event)
            listeners = list(self._subscribers)
        level = {"info": logging.INFO, "notice": logging.INFO,
                 "warning": logging.WARNING, "critical": logging.ERROR}[severity]
        logger.log(level, "%s %s %s", kind, subject, sorted(evidence))
        for fn in listeners:
            try:
                fn(event)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
                # a subscriber must not stop a pairing
                logger.exception("onboard subscriber failed")
        return event

    def recent(self, kind: str = "") -> list[Event]:
        with self._lock:
            return [e for e in self._recent if e.kind.startswith(kind)]


BUS = Bus()
"""The process-wide bus; the servers and clients here emit to it unless given another."""

