"""The headline numbers for a decider on labelled cases, and the rule for "no worse than a
baseline"."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from poolhouse.decide import calibrate
from poolhouse.decide.eval import FLOOR, brier, ece, top_of


@dataclass(frozen=True, slots=True)
class Metrics:
    """Accuracy, Brier (0 to 2), ECE (10 bins), mean confidence and the abstain rate at ``floor``.

    ``answered_accuracy`` is the accuracy over the cases at or above ``floor`` (NaN when none).
    """

    n: int
    accuracy: float
    brier: float
    ece: float
    confidence: float
    floor: float
    abstain_rate: float
    answered_accuracy: float

    def public(self) -> dict[str, Any]:
        """A JSON-ready dict."""
        return {k: getattr(self, k) for k in self.__slots__}

    def line(self) -> str:
        """One line with the headline numbers."""
        return (f"n={self.n} acc={self.accuracy:.3f} brier={self.brier:.3f} ece={self.ece:.3f} "
                f"abstain@{self.floor:g}={self.abstain_rate:.3f}")


def of_rows(rows: Sequence[Sequence[float]], labels: Sequence[int], *,
            floor: float = FLOOR) -> Metrics:
    """`Metrics` for probability rows (option order) and the index of each right answer."""
    if not rows:
        raise ValueError("no rows to score")
    hits = [top_of(r) == y for r, y in zip(rows, labels, strict=True)]
    kept = [ok for r, ok in zip(rows, hits, strict=True) if max(r) >= floor]
    return Metrics(
        n=len(rows), accuracy=sum(hits) / len(hits), brier=brier(rows, labels),
        ece=ece(rows, labels), confidence=sum(max(r) for r in rows) / len(rows), floor=floor,
        abstain_rate=1.0 - len(kept) / len(rows),
        answered_accuracy=sum(kept) / len(kept) if kept else float("nan"))


def worse_than(candidate: Metrics, baseline: Metrics) -> list[str]:
    """Why ``candidate`` is worse than ``baseline``: lower accuracy, higher Brier. Empty when it
    is not."""
    out = []
    if candidate.accuracy < baseline.accuracy:
        out.append(f"accuracy {candidate.accuracy:.3f} is below the baseline's "
                   f"{baseline.accuracy:.3f}")
    if candidate.brier > baseline.brier:
        out.append(f"Brier {candidate.brier:.3f} is above the baseline's {baseline.brier:.3f}")
    return out


def fit_temperature_ece(rows: Sequence[Sequence[float]], labels: Sequence[int], *,
                        low: float = 0.25, high: float = 8.0, steps: int = 80) -> float:
    """The temperature on a log grid that minimises expected calibration error, ties broken
    by NLL."""
    if len(rows) != len(labels):
        raise ValueError(f"{len(rows)} rows but {len(labels)} labels")
    grid = [math.exp(math.log(low) + (math.log(high) - math.log(low)) * i / (steps - 1))
            for i in range(steps)]
    return min(grid, key=lambda t: (round(ece([calibrate.rescale(p, t) for p in rows], labels), 6),
                                    calibrate.nll(rows, labels, t)))
