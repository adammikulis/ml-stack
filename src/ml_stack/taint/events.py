"""Events for a tainted call that was refused or put to the person."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["TaintEvent", "emit", "subscribe"]

logger = logging.getLogger("ml_stack.guard")
logger.addHandler(logging.NullHandler())

_lock = threading.Lock()
_subscribers: list[Callable[[TaintEvent], Any]] = []


@dataclass(frozen=True, slots=True)
class TaintEvent:
    """What happened: ``kind`` (``taint.denied`` or ``taint.confirm``), ``severity``, the
    ``subject`` (``tool_call:<name>``) and ``evidence``, which holds ids, digests and counts and
    never argument text."""

    kind: str
    severity: str
    subject: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    source: str = "taint"
    ts: float = field(default_factory=time.time)


def subscribe(fn: Callable[[TaintEvent], Any]) -> Callable[[], None]:
    """Call ``fn`` with every event from now on; the returned function stops it."""
    with _lock:
        _subscribers.append(fn)

    def stop() -> None:
        with _lock:
            if fn in _subscribers:
                _subscribers.remove(fn)

    return stop


def emit(event: TaintEvent) -> None:
    """Log ``event`` and hand it to each subscriber; a subscriber that raises is logged."""
    logger.warning("%s: %s %s", event.source, event.kind, event.subject)
    with _lock:
        listeners = list(_subscribers)
    for fn in listeners:
        try:
            fn(event)
        except Exception:
            logger.exception("taint subscriber %r failed", fn)
