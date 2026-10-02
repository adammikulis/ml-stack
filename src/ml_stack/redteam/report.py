"""What a red-team run found: its attempts, a summary per target and attack class, a
markdown rendering, and the comparison of two runs."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Attempt", "Report", "compare", "markdown", "summary"]

VERSION = 1


@dataclass(frozen=True, slots=True)
class Attempt:
    """One attack on one target. ``succeeded`` is objective evidence of the attacker's goal
    (a canary touched, a secret read back); ``attempted`` is the model asking for the
    dangerous action; ``blocked`` is a guard refusing it; ``error`` is the target failing to
    answer, which proves nothing either way."""

    target: str
    attack_class: str
    attack_id: str
    succeeded: bool
    arm: str = ""
    attempted: bool = False
    blocked: bool = False
    seconds: float = 0.0
    detail: str = ""
    error: bool = False

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.target, self.attack_class, self.attack_id, self.arm)


@dataclass(slots=True)
class Report:
    """Attempts plus where they came from: model, ml-stack ref, PyRIT version, date."""

    meta: dict[str, Any] = field(default_factory=dict)
    attempts: list[Attempt] = field(default_factory=list)

    def add(self, *attempts: Attempt) -> None:
        self.attempts.extend(attempts)

    def to_json(self) -> str:
        return json.dumps({"version": VERSION, "meta": self.meta,
                           "attempts": [asdict(a) for a in self.attempts]}, indent=1)

    @classmethod
    def from_json(cls, text: str) -> Report:
        data = json.loads(text)
        if data.get("version") != VERSION:
            raise ValueError(f"report version {data.get('version')!r}, expected {VERSION}")
        return cls(dict(data["meta"]), [Attempt(**row) for row in data["attempts"]])

    @classmethod
    def load(cls, path: Path | str) -> Report:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def summary(report: Report) -> list[dict[str, Any]]:
    """One row per (target, attack class, arm): attempts, successes, attempted, blocked,
    errors and the median seconds."""
    groups: dict[tuple[str, str, str], list[Attempt]] = defaultdict(list)
    for one in report.attempts:
        groups[(one.target, one.attack_class, one.arm)].append(one)
    rows = []
    for (target, attack_class, arm), got in sorted(groups.items()):
        rows.append({"target": target, "attack_class": attack_class, "arm": arm,
                     "attempts": len(got), "succeeded": sum(a.succeeded for a in got),
                     "attempted": sum(a.attempted for a in got),
                     "blocked": sum(a.blocked for a in got),
                     "errors": sum(a.error for a in got),
                     "median_s": round(statistics.median(a.seconds for a in got), 2)})
    return rows


def markdown(report: Report) -> str:
    """The report as a document: where it came from, then the table."""
    lines = ["# Red-team report", ""]
    lines += [f"- {key}: {value}" for key, value in sorted(report.meta.items())]
    lines += ["", "| target | attack class | arm | attempts | succeeded | model attempted "
              "| guard blocked | errors | median s |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary(report):
        lines.append(f"| {row['target']} | {row['attack_class']} | {row['arm'] or '-'} "
                     f"| {row['attempts']} | {row['succeeded']} | {row['attempted']} "
                     f"| {row['blocked']} | {row['errors']} | {row['median_s']} |")
    wins = [a for a in report.attempts if a.succeeded]
    if wins:
        lines += ["", "## Attacks that succeeded", ""]
        lines += [f"- `{a.target}` / {a.attack_class} / `{a.attack_id}`"
                  f"{' [' + a.arm + ']' if a.arm else ''}: {a.detail}" for a in wins]
    return "\n".join(lines) + "\n"


def _outcomes(report: Report) -> dict[tuple[str, str, str, str], bool]:
    out: dict[tuple[str, str, str, str], bool] = {}
    for one in report.attempts:
        out[one.key] = out.get(one.key, False) or one.succeeded
    return out


def compare(old: Report, new: Report) -> dict[str, list[tuple[str, str, str, str]]]:
    """Attacks by key: ``regressed`` failed before and succeed now, ``fixed`` the reverse,
    ``new`` and ``gone`` only exist in one run."""
    before, after = _outcomes(old), _outcomes(new)
    return {
        "regressed": sorted(k for k in after if k in before and after[k] and not before[k]),
        "fixed": sorted(k for k in after if k in before and before[k] and not after[k]),
        "new": sorted(k for k in after if k not in before),
        "gone": sorted(k for k in before if k not in after),
    }
