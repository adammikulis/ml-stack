"""A list of callbacks that can be added to and removed from while events are being delivered."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

__all__ = ["Subscribers"]


class Subscribers:
    """The subscription half of an event bus: ``subscribe`` returns the call that ends it, and
    ``listeners`` is a copy to deliver to, taken under the lock."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[Any], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, fn: Callable[[Any], None]) -> Callable[[], None]:
        """Call ``fn`` with every event from now on; returns the call that unsubscribes."""
        with self._lock:
            self._subscribers.append(fn)

        def stop() -> None:
            with self._lock:
                if fn in self._subscribers:
                    self._subscribers.remove(fn)
        return stop
