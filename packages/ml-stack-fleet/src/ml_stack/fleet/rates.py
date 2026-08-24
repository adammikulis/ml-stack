"""How fast each peer is at each kind of work, remembered across runs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

__all__ = ["Rates", "default_path"]

ALPHA = 0.3
"""EWMA weight for a new observation. Low enough that one slow run -- a thermal blip, a"""


def default_path() -> Path:
    return Path(os.environ.get("ML_STACK_RATES")
                or Path.home() / ".ml-stack" / "rates.json").expanduser()


class Rates:
    """Observed units/second, per (peer, kind of work)."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_path()
        self._seen: dict[str, float] = {}
        if self.path.exists():
            try:
                self._seen = {k: float(v) for k, v in
                              json.loads(self.path.read_text()).items()}
            except (OSError, ValueError, TypeError):
                self._seen = {}

    @staticmethod
    def key(peer: str, kind: str) -> str:
        return f"{peer}\t{kind}"

    def get(self, peer: str, kind: str) -> float | None:
        return self._seen.get(self.key(peer, kind))

    def as_map(self) -> dict[tuple[str, str], float]:
        """The shape ``pool.candidates`` wants."""
        out = {}
        for k, v in self._seen.items():
            peer, _, kind = k.partition("\t")
            out[(peer, kind)] = v
        return out

    def record(self, peer: str, kind: str, *, units: float, seconds: float) -> float | None:
        """Fold one completed job in. Returns the new rate, or None if unusable."""
        if units <= 0 or seconds <= 0:
            return None
        observed = units / seconds
        key = self.key(peer, kind)
        prior = self._seen.get(key)
        self._seen[key] = observed if prior is None else (
            ALPHA * observed + (1 - ALPHA) * prior)
        return self._seen[key]

    def save(self) -> Path:
        """Write atomically. Two coordinators finishing at once must not leave a file"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(self._seen, fh, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return self.path

    def __len__(self) -> int:
        return len(self._seen)

    def __repr__(self) -> str:
        return f"Rates({len(self._seen)} measured, at {self.path})"
