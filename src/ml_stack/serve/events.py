"""The progress callback a lease or an escalation reports its steps through, and who is
asking."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Caller", "Event", "Growth", "emit"]

Event = Callable[[dict[str, Any]], None]
"""``on_event({"event": name, ...fields})`` -- a step a caller waiting on a lease or an
escalation can show as it happens, not after. A handler's own errors are swallowed."""


def emit(on_event: Event | None, event: str, **fields: Any) -> None:
    """Hand one step to ``on_event``, swallowing whatever the handler raises."""
    if on_event is None:
        return
    with contextlib.suppress(Exception):
        on_event({"event": event, **fields})


@dataclass(frozen=True)
class Caller:
    """Who is asking a broker for a server, and how to tell them what happens. ``pid`` 0 is
    the process making the call; ``claim`` is what it says about the lease (`provenance.asked`)."""

    pid: int = 0
    label: str = ""
    on_event: Event | None = None
    say: Callable[[str], None] | None = None
    claim: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Growth:
    """How a server's slots are grown: by how many, within what memory and time."""

    add_slots: int = 1
    room: int | None = None
    timeout: float | None = None
    anyway: bool = False
