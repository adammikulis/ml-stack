"""A sliding-window rate limit per sender, shared between processes."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from poolhouse.workspace.chain import held

__all__ = ["RateLimited", "Rates"]



STALE = 64


class RateLimited(RuntimeError):
    """A sender wrote more than the window allows."""


class Rates:
    """Counts each sender's writes in the last ``window_s`` seconds, one file per sender: a
    ``rates v1`` line then one timestamp per write, appended without a sync."""

    def __init__(self, base: Path, most: int, window_s: float,
                 clock: Callable[[], float] = time.time) -> None:
        self.folder = base / "rates"
        self.most, self.window_s, self.clock = most, window_s, clock

    def _file(self, sender: str) -> Path:
        return self.folder / f"{sender.replace('/', '~')}.txt"

    def _times(self, path: Path, now: float) -> list[float]:
        return self._read(path, now)[0]

    def _read(self, path: Path, now: float) -> tuple[list[float], int]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return [], -1
        found = []
        for line in lines[1:]:
            try:
                t = float(line)
            except ValueError:
                continue
            if now - t < self.window_s:
                found.append(t)
        return found, len(lines) - 1

    def recent(self, sender: str) -> int:
        """How many writes ``sender`` made inside the window."""
        return len(self._times(self._file(sender), self.clock()))

    def admit(self, sender: str, most: int = 0) -> None:
        """Count one write by ``sender``; `RateLimited` when it is over ``most`` (default: the
        limit given at construction)."""
        path = self._file(sender)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with held(path.with_name(path.name + ".lock")):
            now = self.clock()
            recent, total = self._read(path, now)
            top = min(most, self.most) if most else self.most
            if len(recent) >= top:
                raise RateLimited(f"{sender} wrote {len(recent)} times in {self.window_s:.0f}s; "
                                  f"the limit is {top}")
            if total < 0 or total - len(recent) > STALE:
                path.write_text("".join(["rates v1\n", *(f"{t}\n" for t in recent)]), encoding="utf-8")
            with path.open("a", encoding="utf-8") as out:
                out.write(f"{now}\n")
