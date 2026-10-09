"""Measures the destructive-action classifier on the labelled corpus and the held-out set."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from poolhouse.decide.guards import destructive as corpus, destructive_adversarial as held_out
from poolhouse.guard.harm import ASKS, Verdict
from poolhouse.interventions import Call

__all__ = ["Report", "Row", "evaluate", "rows", "table"]

Classifier = Callable[[Call, tuple[str, ...]], Verdict]


@dataclass(frozen=True, slots=True)
class Row:
    """One labelled call: the set it is from, its true label and what came back."""

    source: str
    label: str
    call: Call
    got: Verdict
    ms: float


@dataclass(slots=True)
class Report:
    """Recall on destructive calls, false asks on safe and reversible ones, and latency."""

    rows: list[Row] = field(default_factory=list)

    def _of(self, label: str, source: str = "") -> list[Row]:
        return [r for r in self.rows if r.label == label and source in ("", r.source)]

    def recall(self, source: str = "", *, strict: bool = False) -> float:
        """The share of destructive calls labelled destructive (``strict``) or asked about."""
        hit = self._of("destructive", source)
        want = {"destructive"} if strict else ASKS
        return sum(r.got.label in want for r in hit) / len(hit) if hit else 1.0

    def false_asks(self, label: str, source: str = "") -> float:
        """The share of ``label`` calls (safe or reversible) that ask the person."""
        hit = self._of(label, source)
        return sum(r.got.asks for r in hit) / len(hit) if hit else 0.0

    def latency(self) -> tuple[float, float, float]:
        """Mean, 95th percentile and maximum milliseconds per call."""
        ms = sorted(r.ms for r in self.rows)
        return (sum(ms) / len(ms), ms[int(0.95 * (len(ms) - 1))], ms[-1]) if ms else (0.0, 0.0, 0.0)

    def misses(self) -> list[Row]:
        """Destructive calls that were not asked about, and safe or reversible ones that were."""
        return [r for r in self.rows if (r.label == "destructive") != r.got.asks
                and (r.label == "destructive" or r.label in ("safe", "reversible"))]


def rows(source: str) -> Iterable[tuple[str, Call]]:
    """The ``(label, call)`` pairs of ``source``: ``corpus`` or ``held-out``."""
    if source == "corpus":
        for label, pool in (("safe", corpus.SAFE), ("reversible", corpus.REVERSIBLE),
                            ("destructive", corpus.DESTRUCTIVE)):
            for tool, args, _ in pool:
                yield label, Call(tool, args)
    else:
        for label, pool in (("safe", held_out.SAFE), ("reversible", held_out.REVERSIBLE),
                            ("destructive", held_out.DESTRUCTIVE)):
            for tool, args in pool:
                yield label, Call(tool, args)


def evaluate(classify: Classifier) -> Report:
    """Classify every call of both sets, in a project directory holding ``held_out.FILES``."""
    report = Report()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in held_out.FILES:
            (root / name).write_text("x")
        (root / "build").mkdir()
        for source in ("corpus", "held-out"):
            for label, call in rows(source):
                began = time.perf_counter()
                got = classify(call, (str(root),))
                report.rows.append(Row(source, label, call, got,
                                       (time.perf_counter() - began) * 1000))
    return report


def table(name: str, report: Report) -> list[str]:
    """The report as lines: recall, false asks and latency per set."""
    mean, p95, worst = report.latency()
    lines = [f"{name}", f"{'set':10} {'recall':>7} {'strict':>7} {'ask|safe':>9} {'ask|rev':>8}"]
    for source in ("corpus", "held-out"):
        lines.append(f"{source:10} {report.recall(source):7.3f} {report.recall(source, strict=True):7.3f} "
                     f"{report.false_asks('safe', source):9.3f} {report.false_asks('reversible', source):8.3f}")
    lines.append(f"latency ms per call: mean {mean:.3f}  p95 {p95:.3f}  max {worst:.3f}")
    return lines
