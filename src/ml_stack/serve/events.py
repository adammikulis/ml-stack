"""The progress callback a lease or an escalation reports its steps through."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

__all__ = ["Event", "emit"]

Event = Callable[[dict[str, Any]], None]
"""``on_event({"event": name, ...fields})`` -- a step a caller waiting on a lease or an
escalation can show as it happens, not after. A handler's own errors are swallowed."""


def emit(on_event: Event | None, event: str, **fields: Any) -> None:
    """Hand one step to ``on_event``, swallowing whatever the handler raises."""
    if on_event is None:
        return
    with contextlib.suppress(Exception):
        on_event({"event": event, **fields})
