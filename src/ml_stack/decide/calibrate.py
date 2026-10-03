"""Calibration: temperature scaling and isotonic regression on the top probability."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

FLOOR = 1e-12


def softmax(logits: Sequence[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rescale(probs: Sequence[float], temperature: float) -> list[float]:
    """``softmax(log(p) / temperature)``; above 1 flattens, below 1 sharpens."""
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError(f"temperature must be positive and finite, got {temperature}")
    return softmax([math.log(max(p, FLOOR)) / temperature for p in probs])


def nll(rows: Sequence[Sequence[float]], labels: Sequence[int], temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of the labelled options after scaling by ``temperature``."""
    if not rows:
        raise ValueError("no rows to score")
    return -sum(math.log(max(rescale(p, temperature)[y], FLOOR))
                for p, y in zip(rows, labels, strict=True)) / len(rows)


def fit_temperature(rows: Sequence[Sequence[float]], labels: Sequence[int], *,
                    low: float = 0.05, high: float = 20.0) -> float:
    """The temperature minimising NLL on ``rows``, by golden-section search on its log."""
    if len(rows) != len(labels):
        raise ValueError(f"{len(rows)} rows but {len(labels)} labels")
    a, b = math.log(low), math.log(high)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(rows, labels, math.exp(c)), nll(rows, labels, math.exp(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(rows, labels, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(rows, labels, math.exp(d))
    return math.exp((a + b) / 2.0)


BINS = 10


def top_of(row: Sequence[float]) -> int:
    """Index of the largest probability; the first of a tie."""
    return max(range(len(row)), key=lambda i: (row[i], -i))


def reliability(rows: Sequence[Sequence[float]], labels: Sequence[int], bins: int = BINS
                ) -> list[dict[str, float]]:
    """Per confidence bin: its range, how many cases fell in it, mean confidence and accuracy."""
    held: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for row, y in zip(rows, labels, strict=True):
        top = top_of(row)
        held[min(int(row[top] * bins), bins - 1)].append((row[top], top == y))
    return [{"low": i / bins, "high": (i + 1) / bins, "n": len(b),
             "confidence": sum(c for c, _ in b) / len(b),
             "accuracy": sum(ok for _, ok in b) / len(b)} for i, b in enumerate(held) if b]


def ece(rows: Sequence[Sequence[float]], labels: Sequence[int], bins: int = BINS) -> float:
    """Expected calibration error: the bin-weighted gap between confidence and accuracy."""
    total = len(rows)
    return sum(b["n"] / total * abs(b["confidence"] - b["accuracy"])
               for b in reliability(rows, labels, bins)) if total else 0.0


def fit_isotonic(confidences: Sequence[float], correct: Sequence[bool]
                 ) -> list[tuple[float, float]]:
    """Pool-adjacent-violators fit of accuracy against confidence: ``(x, y)`` knots, ascending."""
    if len(confidences) != len(correct):
        raise ValueError(f"{len(confidences)} confidences but {len(correct)} outcomes")
    if not confidences:
        raise ValueError("no rows to fit")
    pairs = sorted(zip(confidences, correct, strict=True))
    blocks: list[list[float]] = []  # [sum of y, count, last x]
    for x, ok in pairs:
        blocks.append([1.0 if ok else 0.0, 1.0, x])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            top = blocks.pop()
            blocks[-1] = [blocks[-1][0] + top[0], blocks[-1][1] + top[1], top[2]]
    return [(b[2], b[0] / b[1]) for b in blocks]


def _interpolate(knots: Sequence[tuple[float, float]], x: float) -> float:
    if x <= knots[0][0]:
        return knots[0][1]
    for (x0, y0), (x1, y1) in itertools.pairwise(knots):
        if x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return knots[-1][1]


@dataclass(frozen=True, slots=True)
class Calibration:
    """A temperature, then optionally an isotonic map of the top probability.

    The isotonic step replaces the chosen option's probability with the accuracy seen at that
    confidence and shares the rest across the others in their original proportions.
    """

    temperature: float = 1.0
    knots: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    n: int = 0

    def apply(self, probs: Sequence[float]) -> list[float]:
        """The calibrated distribution for ``probs``."""
        out = rescale(probs, self.temperature) if self.temperature != 1.0 else list(probs)
        if not self.knots:
            return out
        best = max(range(len(out)), key=lambda i: (out[i], -i))
        top = min(max(_interpolate(self.knots, out[best]), 1.0 / len(out)), 1.0 - 1e-6)
        rest = sum(p for i, p in enumerate(out) if i != best)
        if rest <= 0:
            other = (1.0 - top) / (len(out) - 1)
            return [top if i == best else other for i in range(len(out))]
        scale = (1.0 - top) / rest
        return [top if i == best else p * scale for i, p in enumerate(out)]

    def public(self) -> dict[str, Any]:
        """A JSON-ready dict that `from_public` reads back."""
        return {"temperature": self.temperature, "knots": [list(k) for k in self.knots],
                "n": self.n}

    @classmethod
    def from_public(cls, data: dict[str, Any]) -> Calibration:
        """A `Calibration` from `public()`, validated."""
        t = float(data.get("temperature", 1.0))
        if t <= 0 or not math.isfinite(t):
            raise ValueError(f"temperature must be positive and finite, got {t}")
        knots = tuple((float(x), float(y)) for x, y in data.get("knots", ()))
        return cls(t, knots, int(data.get("n", 0)))


def fit(rows: Sequence[Sequence[float]], labels: Sequence[int], *,
        isotonic: bool = False) -> Calibration:
    """Fit temperature on ``rows`` (probabilities in option order) and, if asked, isotonic."""
    t = fit_temperature(rows, labels)
    cal = Calibration(t, (), len(rows))
    if not isotonic:
        return cal
    scaled = [cal.apply(p) for p in rows]
    tops = [max(range(len(p)), key=lambda i: (p[i], -i)) for p in scaled]
    knots = fit_isotonic([p[i] for p, i in zip(scaled, tops, strict=True)],
                         [i == y for i, y in zip(tops, labels, strict=True)])
    return Calibration(t, tuple(knots), len(rows))
