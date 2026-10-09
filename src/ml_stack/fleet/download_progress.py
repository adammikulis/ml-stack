"""Download rates and console status."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Transfer:
    done: int = 0
    total: int = 0
    speed: float = 0.0
    eta: float | None = None
    _started: float = field(default_factory=time.monotonic)
    _initial: int | None = None

    def update(self, done: int, total: int) -> None:
        now = time.monotonic()
        if self._initial is None or done < self.done:
            self._initial, self._started = done, now
        elapsed = now - self._started
        self.done, self.total = done, total
        self.speed = max(0, done - self._initial) / elapsed if elapsed > 0 else 0
        self.eta = max(0, total - done) / self.speed if total and self.speed else None

    def text(self) -> str:
        amount = f"{self.done / (1 << 20):.1f} MB"
        if self.total:
            amount += f" / {self.total / (1 << 20):.1f} MB"
        if self.speed:
            amount += f" · {self.speed / (1 << 20):.1f} MB/s"
        if self.eta is not None:
            amount += f" · about {self.eta / 60:.1f} min left"
        return amount
