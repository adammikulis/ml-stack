"""How fast each peer is at each kind of work, remembered across runs.

Nothing new is measured to build this: a job already records how long it took
(``Job.elapsed_s``), and a job that says how many units it processed gives the other
half. The quotient, smoothed, is the whole file.

Two decisions worth not reversing by accident:

**Keyed on the peer's name and host, not its beacon instance.** ``instance`` is
regenerated every time the daemon restarts, so keying on it would throw away every
measurement each time a box rebooted -- and a box that reboots is exactly the one you
have the least information about.

**Per kind of work, not one number per peer.** The box that is fastest at training is
not the box that is fastest at tokenizing, and a single "speed" would average those into
a number describing neither.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

__all__ = ["Rates", "default_path"]

ALPHA = 0.3
"""EWMA weight for a new observation. Low enough that one slow run -- a thermal blip, a
noisy neighbour -- does not rewrite a peer's reputation, high enough that a box that has
genuinely changed is believed within a few jobs."""


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
                # A corrupt rates file must not stop a run. Forgetting how fast the
                # boxes are costs one round of exploration; refusing to start costs
                # the whole job.
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
        """Write atomically. Two coordinators finishing at once must not leave a file
        that is half of each."""
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
