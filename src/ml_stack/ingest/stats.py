"""One run's numbers, read off the files beside its store: how far it has got, how fast it
is reading, and what its draft head is doing.

`run_stats` is the one reader. Every call the run kept is folded into a
`ml_stack.client.spent.Spent`, which is where token rates and draft acceptance are already
worked out, so a command formats a `RunStats` rather than adding calls up again.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ml_stack.client.spent import Spent
from ml_stack.home import expand
from ml_stack.ingest.progress import GIVE_UP, Progress
from ml_stack.ingest.sources import Sources, run_attrs
from ml_stack.log import say
from ml_stack.telemetry import Call

__all__ = ["RunStats", "run_stats", "status"]


@dataclass(frozen=True)
class RunStats:
    """What a store's reading has done so far, what it cost, and how the draft head did."""

    out: Path
    started: str = ""
    last_at: str = ""                # when the newest section finished
    sections: int = 0
    wanted: int = 0
    failed: int = 0
    given_up: int = 0
    seconds: float = 0.0             # wall clock across the sections that finished
    spent: Spent = field(default_factory=Spent)
    runs: int = 0                    # runs that wrote into this store
    model: str = ""
    head: str = ""                   # the draft head file, "" when the record names none
    head_depth: int | None = None    # tokens guessed ahead of the model, per pass
    sampling: Mapping[str, Any] = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return max(self.wanted - self.sections, 0)

    @property
    def per_section(self) -> float:
        """Seconds one section took."""
        return self.seconds / self.sections if self.sections else 0.0

    @property
    def left_seconds(self) -> float:
        """Seconds still to read, at the rate this store has measured."""
        return self.remaining * self.per_section

    @property
    def going_seconds(self) -> float:
        """Seconds between the run starting and its newest section finishing."""
        return _between(self.started, self.last_at)

    @property
    def drafting(self) -> bool:
        """Whether a draft head guessed anything for this run."""
        return bool(self.spent.draft_tokens)

    @property
    def passes(self) -> int | None:
        """Verification passes the model made, or None where nothing was drafted."""
        if self.spent.verify_n:
            return self.spent.verify_n
        taken = self.spent.draft_taken
        if not self.spent.completion_tokens or not taken:
            return None
        # where a server does not count its passes: one pass writes one token of its own,
        # so the tokens written less the drafted ones accepted is the number of passes
        return self.spent.completion_tokens - taken

    @property
    def tokens_per_pass(self) -> float | None:
        """Tokens written per verification pass, or None where nothing was drafted."""
        passes = self.passes
        return self.spent.completion_tokens / passes if passes else None

    @property
    def prompt_share(self) -> float | None:
        """Share of the server's time spent reading prompts rather than writing."""
        both = self.spent.prompt_ms + self.spent.predicted_ms
        return self.spent.prompt_ms / both if both else None


def run_stats(out: str | Path, *, view: Any = None) -> RunStats:
    """One store's run, added up: its progress file, its reads, and the run it was read by.

    ``view`` is a `ml_stack.ingest.sources.Sources` a caller already has; one is opened
    where it is not given.
    """
    view = view if view is not None else Sources(out)
    totals = view.progress.totals()
    spent = Spent()
    run_ids: set[str] = set()
    for one in view.sources():
        for row in view.reads(one.slug):
            if run := str(row.get("run") or ""):
                run_ids.add(run)
            for call in row.get("calls") or ():
                if isinstance(call, Mapping):
                    spent.add(Call.from_kept(call))
    attrs = _newest(run_attrs(out, run_ids))
    return RunStats(
        out=expand(out), started=str(totals["started"]), last_at=_last_at(view),
        sections=int(totals["sections"]), wanted=int(totals["of"]),
        failed=int(totals["failed"]), given_up=int(totals["given_up"]),
        seconds=float(totals["seconds"]), spent=spent, runs=len(run_ids),
        model=Path(str(attrs.get("model") or "")).name,
        head=_head_named(str(attrs.get("serving") or "")),
        head_depth=_depth(attrs.get("n_max")),
        sampling=dict(attrs.get("sampling") or {}))


def _last_at(view: Any) -> str:
    """The newest ``at`` any section was written with, or ''."""
    stamps = [str(entry.get("at") or "")
              for one in view.progress.state["sources"].values()
              for entry in (one.get("done") or {}).values()]
    return max(stamps) if stamps else ""


def _newest(runs: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """The run that started last, or an empty record where the store names none."""
    listed = sorted(runs.values(), key=lambda one: str(one.get("started") or ""))
    return listed[-1] if listed else {}


def _head_named(serving: str) -> str:
    """The draft head file a run's serving record names, or ''."""
    found = re.search(r"--draft\s+(\S+)", serving)
    return Path(found.group(1)).name if found else ""


def _depth(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _between(started: str, ended: str) -> float:
    """Seconds between two ``%FT%T`` stamps; 0.0 where either is missing or unreadable."""
    try:
        return (datetime.fromisoformat(ended) - datetime.fromisoformat(started)).total_seconds()
    except (TypeError, ValueError):
        return 0.0


def status(out: str | Path, *, say: Callable[[str], None] = say) -> int:
    """``ml-stack-ingest status``: sections done, what failed, what is folded, and how long left.

    The estimate is the units still to read at the rate this store has actually measured,
    so it is honest about this machine and this model rather than about any other.
    """
    where = Progress.beside(out)
    if not where.is_file():
        say(f"nothing ingested into {out}: no {where.name}")
        return 1
    progress = Progress(where)
    totals = progress.totals()
    say(f"{out}: {totals['sections']} of {totals['of']} sections in "
        f"{totals['sources']} source(s), started {totals['started']}")
    left = 0.0
    read = Sources(out)
    per_source = {s.slug: s for s in read.sources()}
    for slug, one in sorted(progress.state["sources"].items()):
        entries = one.get("done") or {}
        spent = sum(float(e.get("seconds") or 0.0) for e in entries.values())
        broke = sum(1 for e in entries.values() if e.get("error"))
        wanted = int(one.get("sections") or 0)
        rate = spent / len(entries) if entries else totals["per_section"]
        remaining = max(wanted - len(entries), 0) * rate
        left += remaining
        say(f"  {slug:<28} {len(entries):>4} / {one.get('sections') or '?':<5} "
            f"{spent / 60:6.1f} min" + (f"  {broke} failed" if broke else "")
            + (f"  ~{_for_long(remaining)} left" if remaining else ""))
        held = per_source.get(slug)
        if held is not None and held.units:
            say(f"      cost: {held.prompt_tokens:,} read + {held.completion_tokens:,} written "
                f"token(s) over {held.units} unit(s); {held.per_unit:.1f} s/unit, "
                f"{held.tokens_per_unit:,} tokens/unit")
        folded = int(one.get("folded_at") or 0)
        if folded:
            say(f"      in store: {int(one.get('folded_nodes') or 0)} nodes, "
                f"{int(one.get('folded_edges') or 0)} edges, "
                f"folded at unit {folded} of {wanted or '?'}"
                + (f", in {one['folded_seconds']:.1f}s" if one.get("folded_seconds") else ""))
        elif _source_in_store(out, slug):
            say("      in store: folded by an earlier run (units unknown)")
        else:
            say("      in store: nothing folded yet")
    if totals["sections"]:
        rate = (f", {3600 / totals['per_section']:.0f} sections/hour"
                if totals["per_section"] else "")
        say(f"  {totals['per_section']:.1f} s/section{rate}; "
            f"{totals['seconds'] / 3600:.1f} h spent"
            + (f", ~{_for_long(left)} left" if left else "")
            + (f", {totals['failed']} failed" if totals["failed"] else "")
            + (f" ({totals['given_up']} given up after {GIVE_UP} tries; the reply each wrote "
               f"is `raw` in the reads file)" if totals["given_up"] else ""))
    units = sum(s.units for s in per_source.values())
    if units:
        prompt = sum(s.prompt_tokens for s in per_source.values())
        written = sum(s.completion_tokens for s in per_source.values())
        seconds = sum(s.seconds for s in per_source.values())
        say(f"  total: {prompt:,} read + {written:,} written token(s) over {units} unit(s); "
            f"{seconds / units:.1f} s/unit, {round((prompt + written) / units):,} tokens/unit")
    for line in _cost_said(run_stats(out, view=read)):
        say(line)
    return 0


def _cost_said(stats: Any) -> list[str]:
    """The model, the speed, the drafting and the time, for `status`."""
    out = [f"  model     {stats.model or 'not named by the run record'}"
           + (f"; asked at {_asked_said(stats.sampling)}" if stats.sampling else "")]
    if not stats.spent.calls:
        return [*out, "  speed     no call reached the server, so nothing was measured"]
    return [*out, *_speed_said(stats), *_drafting_said(stats),
            f"  going     {_for_long(stats.going_seconds)} from the start "
            f"({stats.started}) to the newest section ({stats.last_at})"
            + (f"; ~{_for_long(stats.left_seconds)} left at this rate"
               if stats.left_seconds else "")]


def _asked_said(sampling: Mapping[str, Any]) -> str:
    return ", ".join(f"{name} {value}" for name, value in sorted(sampling.items()))


def _speed_said(stats: Any) -> list[str]:
    """How fast the model wrote and read, and which half of the time went where."""
    spent = stats.spent
    wrote, red = spent.decode_tokens_per_second, spent.prompt_tokens_per_second
    if wrote is None:
        return ["  speed     the server reported no generation time"]
    out = [f"  speed     {wrote:.1f} tokens/second written"
           + (f", {red:,.0f} tokens/second read" if red else "")
           + " (higher is better)"]
    if (share := stats.prompt_share) is not None:
        out.append(f"            {spent.predicted_ms / 60000:.1f} min writing, "
                   f"{spent.prompt_ms / 60000:.1f} min reading prompts "
                   f"-- {share * 100:.0f}% of the server's time went on reading")
    return out


def _drafting_said(stats: Any) -> list[str]:
    """What the draft head guessed and how much of it the model kept."""
    if not stats.drafting:
        return ["  drafting  no draft head: every token was written by the model itself"]
    depth = (f"{stats.head_depth} token(s) ahead per pass" if stats.head_depth
             else "depth not named by the run record")
    kept = (f"            {(stats.spent.acceptance or 0) * 100:.1f}% of drafted tokens kept "
            f"(higher is better), {stats.tokens_per_pass:.1f} tokens per verification pass")
    if stats.spent.draft_ms:
        kept += (f"\n            {stats.spent.draft_ms / 1000:.0f}s guessing, "
                 f"{(stats.spent.verify_ms or 0) / 1000:.0f}s checking")
    return [f"  drafting  {stats.head or 'head not named by the run record'}, {depth}", kept]


def _for_long(seconds: float) -> str:
    """A duration a person reads at a glance."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _source_in_store(out: str | Path, slug: str) -> bool:
    """Whether the store holds ``source:<slug>`` -- written by a run before folds were
    recorded, so the progress file says nothing about it."""
    from ml_stack.graph.store import GraphStore

    if not Path(out).expanduser().exists():
        return False
    try:
        with GraphStore(out, read_only=True) as store:
            return any(n["id"] == f"source:{slug}" for n in store.nodes(kind="source"))
    except Exception:  # noqa: BLE001 - a store a writer holds, or none; say nothing
        return False
