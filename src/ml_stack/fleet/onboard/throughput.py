"""How fast a peer really is, and how fast it may be asked to go.

`Rolling` is the measure: bytes completed in the last ``window_s`` seconds, as a rate. A transfer
uses one over all its peers (below the floor for a whole window, the Hub is used instead) and each
peer has one of its own (its rate is kept in the peer book and ranks it for the next pull).
`Limiter` is the optional cap on a peer's bytes per second, the metered-connection / bandwidth
setting (``ml-stack cluster peers limit``): requests are scheduled so that the average never
exceeds it. Both take a clock so tests can move time.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

__all__ = ["Limiter", "Rolling"]


class Rolling:
    """Bytes completed lately. ``rate()`` is None until a whole ``window_s`` has passed since
    the first look, so a slow start is not mistaken for a slow peer."""

    def __init__(self, window_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.window_s, self.clock = window_s, clock
        self._began = clock()
        self._done: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def add(self, nbytes: int) -> None:
        with self._lock:
            self._done.append((self.clock(), nbytes))

    def rate(self) -> float | None:
        """Bytes per second over the last window, or None while the window is not yet full."""
        now = self.clock()
        if now - self._began < self.window_s:
            return None
        with self._lock:
            while self._done and now - self._done[0][0] > self.window_s:
                self._done.popleft()
            return sum(n for _, n in self._done) / self.window_s


class Limiter:
    """At most ``bytes_per_s`` on average: each request books its slot, in order, and waits for it."""

    def __init__(self, bytes_per_s: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if bytes_per_s <= 0:
            raise ValueError("a limit is more than zero bytes a second")
        self.bytes_per_s, self.clock, self._sleep = bytes_per_s, clock, sleep
        self._free_at = 0.0
        self._lock = threading.Lock()

    def wait(self, nbytes: int, cancelled: Callable[[], bool] = lambda: False) -> None:
        """Block until ``nbytes`` may be asked for; returns early when ``cancelled()``."""
        with self._lock:
            now = self.clock()
            start = max(now, self._free_at)
            self._free_at = start + nbytes / self.bytes_per_s
        while (left := start - self.clock()) > 0 and not cancelled():
            self._sleep(min(left, 0.2))
