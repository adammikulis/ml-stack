"""A sliding-window rate limit per sender, shared between processes."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from ml_stack.files import read_json, write_json
from ml_stack.workspace.chain import held

__all__ = ["RateLimited", "Rates"]

VERSION = 1


class RateLimited(RuntimeError):
    """A sender wrote more than the window allows."""


class Rates:
    """Counts each sender's writes in the last ``window_s`` seconds."""

    def __init__(self, base: Path, most: int, window_s: float,
                 clock: Callable[[], float] = time.time) -> None:
        self.path = base / "rates.json"
        self.most, self.window_s, self.clock = most, window_s, clock

    def recent(self, sender: str) -> int:
        """How many writes ``sender`` made inside the window."""
        data = read_json(self.path, {})
        table = data.get("senders", {}) if isinstance(data, dict) else {}
        return sum(1 for t in table.get(sender, []) if self.clock() - t < self.window_s)

    def admit(self, sender: str, most: int = 0) -> None:
        """Count one write by ``sender``; `RateLimited` when it is over ``most`` (default: the
        limit given at construction)."""
        with held(self.path.with_name("rates.lock")):
            now = self.clock()
            data = read_json(self.path, {})
            table = data.get("senders", {}) if isinstance(data, dict) else {}
            table = {k: [t for t in v if now - t < self.window_s] for k, v in table.items()}
            table = {k: v for k, v in table.items() if v}
            recent = table.get(sender, [])
            top = min(most, self.most) if most else self.most
            if len(recent) >= top:
                raise RateLimited(f"{sender} wrote {len(recent)} times in {self.window_s:.0f}s; "
                                  f"the limit is {top}")
            table[sender] = [*recent, now]
            write_json(self.path, {"version": VERSION, "senders": table}, indent=None)
