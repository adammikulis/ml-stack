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
    """Counts each sender's writes in the last ``window_s`` seconds, one file per sender."""

    def __init__(self, base: Path, most: int, window_s: float,
                 clock: Callable[[], float] = time.time) -> None:
        self.folder = base / "rates"
        self.most, self.window_s, self.clock = most, window_s, clock

    def _file(self, sender: str) -> Path:
        return self.folder / f"{sender.replace('/', '~')}.json"

    def _times(self, sender: str, now: float) -> list[float]:
        data = read_json(self._file(sender), {})
        found = data.get("times", []) if isinstance(data, dict) else []
        return [float(t) for t in found if now - float(t) < self.window_s]

    def recent(self, sender: str) -> int:
        """How many writes ``sender`` made inside the window."""
        return len(self._times(sender, self.clock()))

    def admit(self, sender: str, most: int = 0) -> None:
        """Count one write by ``sender``; `RateLimited` when it is over ``most`` (default: the
        limit given at construction)."""
        path = self._file(sender)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with held(path.with_name(path.name + ".lock")):
            now = self.clock()
            recent = self._times(sender, now)
            top = min(most, self.most) if most else self.most
            if len(recent) >= top:
                raise RateLimited(f"{sender} wrote {len(recent)} times in {self.window_s:.0f}s; "
                                  f"the limit is {top}")
            write_json(path, {"version": VERSION, "times": [*recent, now]}, indent=None)
