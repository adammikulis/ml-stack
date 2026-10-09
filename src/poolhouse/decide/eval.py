"""Scoring a decider: accuracy, Brier score, ECE with reliability bins, abstention curves and
latency percentiles."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from poolhouse.decide.base import Decider
from poolhouse.decide.calibrate import (  # noqa: F401  (moved there; still importable from here)
    BINS,
    ece,
    reliability,
    top_of,
)
from poolhouse.decide.cases import Case
from poolhouse.decide.types import DecideError, Decision

FLOOR = 0.8
"""The confidence under which an answer counts as abstained unless the caller says otherwise."""


def brier(rows: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Mean over cases of the sum over options of ``(p - outcome) ** 2``; 0 is perfect, 2 is worst."""
    if not rows:
        raise ValueError("no rows to score")
    return sum(sum((p - (1.0 if k == y else 0.0)) ** 2 for k, p in enumerate(row))
               for row, y in zip(rows, labels, strict=True)) / len(rows)


def abstention_curve(rows: Sequence[Sequence[float]], labels: Sequence[int],
                     points: Sequence[float] | None = None) -> list[dict[str, float]]:
    """For each confidence threshold: the share of cases answered and their accuracy."""
    cuts = list(points) if points is not None else [i / 20 for i in range(0, 20)] + [0.99]
    curve = []
    for cut in cuts:
        kept = [(top_of(r) == y) for r, y in zip(rows, labels, strict=True) if max(r) >= cut]
        curve.append({"threshold": cut, "coverage": len(kept) / len(rows) if rows else 0.0,
                      "accuracy": sum(kept) / len(kept) if kept else float("nan")})
    return curve


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q`` quantile (0 to 1) by linear interpolation; NaN of nothing."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass(slots=True)
class Report:
    """What `evaluate` measured; ``errors`` counts cases the decider failed to answer."""

    backend: str
    model: str
    n: int
    errors: int
    accuracy: float
    brier: float
    ece: float
    nll: float
    mean_confidence: float
    latency_ms: dict[str, float]
    reliability: list[dict[str, float]]
    abstention: list[dict[str, float]]
    by_tag: dict[str, dict[str, float]] = field(default_factory=dict)
    floor: float = FLOOR
    abstain_rate: float = 0.0

    def public(self) -> dict[str, Any]:
        """A JSON-ready dict."""
        return {k: getattr(self, k) for k in self.__slots__}

    def line(self) -> str:
        """One line: the headline numbers."""
        return (f"{self.backend}:{self.model} n={self.n} err={self.errors} "
                f"acc={self.accuracy:.3f} brier={self.brier:.3f} ece={self.ece:.3f} "
                f"abstain@{self.floor:g}={self.abstain_rate:.3f} "
                f"p50={self.latency_ms['p50']:.1f}ms p95={self.latency_ms['p95']:.1f}ms")


def score(decisions: Sequence[Decision], cases: Sequence[Case], *, errors: int = 0,
          floor: float = FLOOR) -> Report:
    """A `Report` for decisions already made, one per case in order."""
    if not decisions:
        raise DecideError("nothing was answered; there is nothing to score")
    rows = [[d.scores[o.name] for o in c.options] for d, c in zip(decisions, cases, strict=True)]
    labels = [c.label_index for c in cases]
    hits = [top_of(r) == y for r, y in zip(rows, labels, strict=True)]
    lat = [d.latency_ms for d in decisions]
    tags: dict[str, list[bool]] = {}
    for c, ok in zip(cases, hits, strict=True):
        for t in c.tags:
            tags.setdefault(t, []).append(ok)
    return Report(
        backend=decisions[0].backend, model=decisions[0].model, n=len(decisions), errors=errors,
        accuracy=sum(hits) / len(hits), brier=brier(rows, labels), ece=ece(rows, labels),
        nll=-sum(math.log(max(r[y], 1e-12)) for r, y in zip(rows, labels, strict=True)) / len(rows),
        mean_confidence=sum(max(r) for r in rows) / len(rows),
        latency_ms={"mean": sum(lat) / len(lat), "p50": percentile(lat, 0.5),
                    "p95": percentile(lat, 0.95), "p99": percentile(lat, 0.99)},
        reliability=reliability(rows, labels), abstention=abstention_curve(rows, labels),
        by_tag={t: {"n": len(v), "accuracy": sum(v) / len(v)} for t, v in sorted(tags.items())},
        floor=floor, abstain_rate=sum(max(r) < floor for r in rows) / len(rows))


def evaluate(decider: Decider, cases: Sequence[Case], *, floor: float = FLOOR) -> Report:
    """Run ``decider`` over ``cases`` one at a time and score it.

    A case the decider raises `DecideError` on is counted in ``errors`` and left out of the
    metrics, so a backend that fails often cannot look accurate.
    """
    made: list[Decision] = []
    kept: list[Case] = []
    for case in cases:
        try:
            made.append(decider.decide(case.question, case.state, case.options))
            kept.append(case)
        except DecideError:
            continue
    return score(made, kept, errors=len(cases) - len(kept), floor=floor)
