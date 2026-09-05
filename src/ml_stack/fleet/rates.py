"""How fast each peer is at each kind of work, remembered across runs."""

from __future__ import annotations

from pathlib import Path

from ml_stack import home
from ml_stack.records import Document

__all__ = ["Rates", "default_path"]

ALPHA = 0.3
"""EWMA weight for a new observation. Low enough that one slow run -- a thermal blip, a"""


_DOC: Document[dict[str, float]] = Document(
    default=lambda: home.state("rates.json"), env="ML_STACK_RATES",
    build=lambda held: {str(k): float(v) for k, v in held.items()},
    unbuild=lambda seen: dict(sorted(seen.items())), empty=dict)


def default_path() -> Path:
    """Where this machine keeps its measured rates. ``ML_STACK_RATES`` moves it."""
    return _DOC.path()


class Rates:
    """Observed units/second, per (peer, kind of work)."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = _DOC.path(path)
        self._seen: dict[str, float] = _DOC.read(self.path)

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
        """Write atomically, sorted, and return where they went."""
        return _DOC.write(self._seen, self.path, atomic=True)

    def __len__(self) -> int:
        return len(self._seen)

    def __repr__(self) -> str:
        return f"Rates({len(self._seen)} measured, at {self.path})"
