"""Which admission class a scripts/test run belongs to: decided by its estimated duration, not by names."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import testhistory
from testslots_policy import BACKGROUND, INTERACTIVE

FALLBACK_TIERS = {"full": "the full tier", "slow": "the slow tier", "record": "the testmon map build"}


@dataclass
class Verdict:
    klass: str
    reason: str
    estimate_s: float | None = None
    recorded: int = 0


@dataclass
class Basis:
    """What an estimate rests on: the recorded durations and the workers the run is expected to get."""
    tests: dict[str, dict]
    workers: int = 1


def selectors(arguments: list[str]) -> list[str]:
    """The test paths and node ids among the pytest arguments."""
    config = pytest.Config.fromdictargs({}, arguments)
    try:
        return list(config.args)
    finally:
        config._ensure_unconfigure()


def explicit(chosen: list[str], root: Path) -> bool:
    """Whether the selectors name a test file or a node id (a bare directory is not explicit)."""
    return any("::" in s or (root / s.split("::", 1)[0]).is_file() for s in chosen)


def fallback(tier: str, rest: list[str], chosen: list[str], root: Path) -> tuple[str, str]:
    """The class from the shape of the command, used while the duration history is too small to estimate from."""
    if "--redteam" in rest:
        return BACKGROUND, "--redteam attacks run for a long time"
    if tier in FALLBACK_TIERS:
        return BACKGROUND, FALLBACK_TIERS[tier]
    if tier in ("fast", "all") and not explicit(chosen, root):
        return BACKGROUND, f"{tier} with no test file or node selector"
    return INTERACTIVE, ""


def classify(tier: str, rest: list[str], root: Path, basis: Basis, background: bool = False) -> Verdict:
    """The class of a run: its estimated wall time against the threshold once enough history exists."""
    if background:
        return Verdict(BACKGROUND, "--background was given")
    if tier == "record":
        return Verdict(BACKGROUND, FALLBACK_TIERS[tier])
    if tier in ("quick", "gate"):
        return Verdict(INTERACTIVE, "")
    chosen = selectors(rest)
    if len(basis.tests) < testhistory.MIN_RECORDED:
        klass, why = fallback(tier, rest, chosen, root)
        return Verdict(klass, why and f"{why}; no duration history yet")
    found = testhistory.estimate(basis.tests, root, chosen)
    wall = found.wall(basis.workers)
    limit = testhistory.threshold()
    klass = BACKGROUND if wall >= limit else INTERACTIVE
    return Verdict(klass, f"estimated {span(wall)} against the {span(limit)} threshold" if klass == BACKGROUND else "",
                   wall, found.recorded)


def span(seconds: float) -> str:
    """Seconds written as "38 s" or "42 min"."""
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.0f} min"


def estimate_line(verdict: Verdict) -> str:
    """The line printed on every run that has an estimate, or an empty string."""
    if verdict.estimate_s is None:
        return ""
    base = f"test: estimated {span(verdict.estimate_s)}"
    return f"{base} from {verdict.recorded:,} recorded tests" if verdict.estimate_s >= 120 else base


def notice(verdict: Verdict) -> str:
    """The one line printed for a background run, or an empty string."""
    if verdict.klass != BACKGROUND:
        return ""
    return (f"test: classed background ({verdict.reason}); interactive runs are admitted first "
            "and this run's CPU priority is lowered")
