"""JevBench's public items as decision cases, and a scored run of the deciders over them.

JevBench (https://github.com/fstandhartinger/jevbench, MIT, Benchmark Heaven; not affiliated with
TypeSafe AI) scores typed decision models on ``noul``, ``choice`` and ``score`` questions. Only
its 231 public items are here, pinned by commit and SHA-256 and fetched through the net pipeline.
Opt-in: no default test tier runs it, and a run uses the GPU. An item the decider could not
answer counts as wrong (JevBench's rule; ``eval.evaluate`` leaves it out). Commands are in
docs/decision-models.md.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack import home, net
from ml_stack.decide.cases import Case
from ml_stack.decide.eval import brier, ece, percentile, top_of
from ml_stack.decide.rules import RulesDecider
from ml_stack.decide.types import BackendUnavailable, DecideError, Decision, Option
from ml_stack.files import sha256_file

REPO = "fstandhartinger/jevbench"
COMMIT = "bb05a335bc809e61b20c0f745d25499a82b326fc"
HOST_URL = "https://raw.githubusercontent.com"
PURPOSE = "jevbench public items"
TYPES = ("noul", "choice", "score")


@dataclass(frozen=True, slots=True)
class Part:
    """One public file of the benchmark: its tier, size and hash at the pinned commit."""

    tier: str
    size: int
    sha256: str

    @property
    def path(self) -> str:
        """Where the file sits in the repository."""
        return f"datasets/public/{self.tier}.jsonl"


PARTS = (
    Part("easy", 37220, "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b"),
    Part("original", 57237, "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"),
    Part("hard", 651848, "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb"),
)


def where(part: Part) -> Path:
    """The cached copy of ``part`` under the pinned commit."""
    return home.cache("decide", "jevbench", COMMIT, f"{part.tier}.jsonl")


def locate(part: Part, *, download: bool = False) -> Path:
    """The verified file for ``part``; downloaded (through the net pipeline, size and hash
    pinned) only when ``download`` is set."""
    path = where(part)
    if not path.is_file():
        if not download:
            raise BackendUnavailable(
                f"{part.path} is not downloaded; run `ml-stack-decide jevbench --fetch`")
        want = net.Want(sha256=part.sha256, size=part.size, require_digest=True,
                        max_bytes=part.size + 4096, purpose=PURPOSE)
        try:
            net.download(f"{HOST_URL}/{REPO}/{COMMIT}/{part.path}", path, want)
        except net.Blocked as exc:
            raise DecideError(f"{part.path} does not match its pinned hash; held: {exc}") from None
        except OSError as exc:
            raise DecideError(f"{part.path} could not be fetched: {exc}") from None
    if path.stat().st_size != part.size or sha256_file(path) != part.sha256:
        path.unlink(missing_ok=True)
        raise DecideError(f"{part.path} does not match its pinned size and hash; removed")
    return path


def _described(row: Mapping[str, Any]) -> tuple[Option, ...]:
    """The options of one item, each with the rubric text the item gives it."""
    kind = row["question"]["type"]
    crit = row["question"].get("criteria")
    labels = [str(x) for x in row["labels"]]
    if kind == "noul":
        crit = crit if isinstance(crit, Mapping) else {}
        say = {"no": crit.get("false", ""), "yes": crit.get("true", "")}
        return tuple(Option(name, str(say.get(name, ""))) for name in labels)
    if isinstance(crit, Mapping):
        return tuple(Option(name, str(crit.get(name, ""))) for name in labels)
    if isinstance(crit, Sequence) and not isinstance(crit, str) and len(crit) == len(labels):
        return tuple(Option(name, str(text)) for name, text in zip(labels, crit, strict=True))
    return tuple(Option(name, "") for name in labels)


def case_of(row: Mapping[str, Any], tier: str) -> Case:
    """One JevBench record as a `Case`. Tags: ``type:``, ``tier:`` and ``family:``. An item
    with no label (``expected`` null) is refused: it cannot be scored."""
    kind = row["question"]["type"]
    if kind not in TYPES:
        raise ValueError(f"{row.get('id')}: unknown question type {kind!r}")
    if row.get("expected") is None:
        raise ValueError(f"{row.get('id')}: no expected answer")
    state = row["state"]
    if not isinstance(state, str):
        state = json.dumps(state, sort_keys=True, ensure_ascii=False)
    return Case(question=str(row["question"]["instructions"]), state=state,
                options=_described(row), label=str(row["expected"]), id=str(row["id"]),
                group=str(row.get("group") or row["id"]),
                tags=(f"type:{kind}", f"tier:{tier}", f"family:{row.get('family', '')}"))


def read_part(path: Path | str, tier: str) -> list[Case]:
    """Every item of one JevBench JSONL file as cases."""
    with Path(path).open(encoding="utf-8") as handle:
        return [case_of(json.loads(line), tier) for line in handle if line.strip()]


def load(tiers: Iterable[str] = (), *, download: bool = False) -> list[Case]:
    """The public items (all tiers, or the ones named) from the pinned files."""
    chosen = [p for p in PARTS if not tiers or p.tier in set(tiers)]
    return [c for p in chosen for c in read_part(locate(p, download=download), p.tier)]


# -- running and scoring ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Outcome:
    """One item's result: the decision, or ``None`` and why not."""

    case: Case
    decision: Decision | None
    error: str = ""


def first_option_rules(case: Case) -> RulesDecider:
    """The rules baseline: no rules, so the first listed option wins every time. For noul that
    is ``no``; it is a floor, not a competitor."""
    return RulesDecider((), default=case.options[0].name)


def run(decide: Callable[[Case], Decision], cases: Sequence[Case],
        progress: Callable[[int, int], None] | None = None) -> list[Outcome]:
    """Ask every case; an answer that fails is kept as an error, never dropped."""
    out = []
    for i, case in enumerate(cases, 1):
        try:
            out.append(Outcome(case, decide(case)))
        except DecideError as exc:
            out.append(Outcome(case, None, str(exc)[:200]))
        if progress:
            progress(i, len(cases))
    return out


def _tag(case: Case, prefix: str) -> str:
    return next((t[len(prefix):] for t in case.tags if t.startswith(prefix)), "")


def metrics(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    """Accuracy over every item (an unanswered one is wrong), Brier and ECE over the answered
    ones, the ordinal error for score items, and latency."""
    answered = [o for o in outcomes if o.decision is not None]
    rows = [[o.decision.scores[opt.name] for opt in o.case.options]  # type: ignore[union-attr]
            for o in answered]
    labels = [o.case.label_index for o in answered]
    hits = sum(top_of(r) == y for r, y in zip(rows, labels, strict=True))
    lat = [o.decision.latency_ms for o in answered]  # type: ignore[union-attr]
    ordinal = [abs(sum(i * p for i, p in enumerate(r)) - y)
               for r, y, o in zip(rows, labels, answered, strict=True)
               if _tag(o.case, "type:") == "score"]
    return {
        "n": len(outcomes), "errors": len(outcomes) - len(answered),
        "accuracy": hits / len(outcomes) if outcomes else math.nan,
        "correct": hits,
        "brier": brier(rows, labels) if rows else math.nan,
        "ece": ece(rows, labels) if rows else math.nan,
        "mean_confidence": statistics.fmean(max(r) for r in rows) if rows else math.nan,
        "chance": statistics.fmean(1 / len(o.case.options) for o in outcomes) if outcomes
        else math.nan,
        "ordinal_mae": statistics.fmean(ordinal) if ordinal else None,
        "p50_ms": percentile(lat, 0.5) if lat else math.nan,
    }


def summarize(outcomes: Sequence[Outcome]) -> dict[str, dict[str, Any]]:
    """``metrics`` for everything, then per question type and per tier."""
    groups: dict[str, list[Outcome]] = defaultdict(list)
    for o in outcomes:
        groups["all"].append(o)
        groups["type:" + _tag(o.case, "type:")].append(o)
        groups["tier:" + _tag(o.case, "tier:")].append(o)
    return {k: metrics(v) for k, v in sorted(groups.items())}


def table(name: str, summary: Mapping[str, Mapping[str, Any]]) -> str:
    """The summary as aligned text."""
    lines = [f"{name}", f"  {'group':<14}{'n':>5}{'err':>5}{'acc':>7}{'chance':>8}{'brier':>7}"
             f"{'ece':>7}{'p50 ms':>9}"]
    for key, m in summary.items():
        lines.append(f"  {key:<14}{m['n']:>5}{m['errors']:>5}{m['accuracy']:>7.3f}"
                     f"{m['chance']:>8.3f}{m['brier']:>7.3f}{m['ece']:>7.3f}{m['p50_ms']:>9.1f}")
    return "\n".join(lines)
