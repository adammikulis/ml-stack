"""What a llama-server counted about its draft head, and the difference across a block of work.

The server keeps four cumulative counters on ``/metrics``: verification steps, draft tokens
offered, draft tokens accepted, and accepted tokens broken down by position within the
draft. Reading them either side of a piece of work gives that work's speculative behaviour
without the client having to be one of ours, so a run through lm-eval counts the same way an
ingest does.

Every block records the workload it was and the sampling it ran under, because acceptance
at temperature 0 measures agreement with the target's top choice and acceptance under
sampling measures how far two distributions sit apart. The two are not the same quantity
and a number without its regime cannot be read.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import monotonic

from ml_stack.http import ServerError, request_bytes

__all__ = ["Block", "Ledger", "Speculative", "counting", "read_speculative", "sampling_named"]

METRICS_PATH = "/metrics"

# The counter names llama.cpp writes, and the field each fills.
_TOTALS = {
    "llamacpp:spec_decode_num_drafts_total": "drafts",
    "llamacpp:spec_decode_num_draft_tokens_total": "drafted",
    "llamacpp:spec_decode_num_accepted_tokens_total": "accepted",
}
_PER_POS = "llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
_PROCESSING = "llamacpp:requests_processing"

GREEDY = "greedy"


def sampling_named(temperature: float | None, *, top_p: float | None = None) -> str:
    """A short name for a sampling regime: `GREEDY` at temperature 0, else ``t<temp>``."""
    if temperature is None:
        return "unsaid"
    if temperature <= 0:
        return GREEDY
    name = f"t{temperature:g}"
    return f"{name}/p{top_p:g}" if top_p is not None and top_p < 1 else name


@dataclass(frozen=True, slots=True)
class Speculative:
    """One reading of a server's speculative counters."""

    drafts: int = 0
    drafted: int = 0
    accepted: int = 0
    per_position: tuple[int, ...] = ()
    processing: int = 0

    def __sub__(self, other: Speculative) -> Speculative:
        width = max(len(self.per_position), len(other.per_position))
        return Speculative(
            drafts=self.drafts - other.drafts,
            drafted=self.drafted - other.drafted,
            accepted=self.accepted - other.accepted,
            per_position=tuple(_at(self.per_position, i) - _at(other.per_position, i)
                               for i in range(width)),
            processing=self.processing,
        )

    @property
    def negative(self) -> bool:
        """Whether any count went backwards, which only a restarted server does."""
        return (self.drafts < 0 or self.drafted < 0 or self.accepted < 0
                or any(n < 0 for n in self.per_position))

    @property
    def acceptance(self) -> float | None:
        """Share of offered draft tokens the model kept."""
        return self.accepted / self.drafted if self.drafted else None

    @property
    def depth_offered(self) -> float | None:
        """Draft tokens offered per verification step."""
        return self.drafted / self.drafts if self.drafts else None

    @property
    def tokens_per_draft(self) -> float | None:
        """Tokens the model got out of one verification step, the free one included."""
        return 1.0 + self.accepted / self.drafts if self.drafts else None

    @property
    def survival(self) -> tuple[float, ...]:
        """Share of drafts whose token at each position survived, position 0 first."""
        if not self.drafts:
            return ()
        return tuple(n / self.drafts for n in self.per_position)

    def public(self) -> dict[str, object]:
        """The counts and the rates, JSON-ready."""
        return {"drafts": self.drafts, "drafted": self.drafted, "accepted": self.accepted,
                "acceptance": _round(self.acceptance, 4),
                "depth_offered": _round(self.depth_offered, 3),
                "tokens_per_draft": _round(self.tokens_per_draft, 3),
                "per_position": list(self.per_position),
                "survival": [round(v, 4) for v in self.survival]}


def _round(value: float | None, places: int) -> float | None:
    return None if value is None else round(value, places)


def read_speculative(base_url: str, *, timeout: float = 5.0) -> Speculative | None:
    """The server's speculative counters, or None where it does not report them.

    None covers all three ways there is nothing to read: ``--metrics`` not passed, a build
    older than the counters, and a server with no draft head.
    """
    try:
        reply = request_bytes(f"{base_url.rstrip('/')}{METRICS_PATH}", timeout=timeout)
    except (ServerError, OSError):
        return None
    return _parse(reply.body.decode("utf-8", "replace"))


def _parse(text: str) -> Speculative | None:
    totals: dict[str, int] = {}
    positions: dict[int, int] = {}
    processing = 0
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.partition(" ")
        if field_name := _TOTALS.get(name):
            totals[field_name] = _int(value)
        elif name == _PROCESSING:
            processing = _int(value)
        elif name.startswith(_PER_POS + "{") and (at := _position_of(name)) is not None:
            positions[at] = _int(value)
    if not totals:
        return None
    per_pos = tuple(positions.get(i, 0) for i in range(max(positions, default=-1) + 1))
    return Speculative(drafts=totals.get("drafts", 0), drafted=totals.get("drafted", 0),
                       accepted=totals.get("accepted", 0), per_position=per_pos,
                       processing=processing)


def _position_of(name: str) -> int | None:
    """The ``position`` label of one per-position series, or None where there is not one."""
    inside = name[name.find("{") + 1:name.rfind("}")]
    for part in inside.split(","):
        key, _, value = part.partition("=")
        if key.strip() != "position":
            continue
        try:
            return int(value.strip().strip('"'))
        except ValueError:
            return None
    return None


def _int(value: str) -> int:
    """A counter's value, which llama.cpp may write as a float; 0 where it is not a number."""
    try:
        return int(float(value.strip()))
    except ValueError:
        return 0


@dataclass
class Block:
    """One piece of work, labelled, with the counters either side of it."""

    workload: str
    sampling: str
    label: str = ""
    before: Speculative | None = None
    after: Speculative | None = None
    seconds: float = 0.0
    note: str = ""

    @property
    def delta(self) -> Speculative | None:
        """What this block alone drafted, or None where it was not measured."""
        if self.before is None or self.after is None:
            return None
        moved = self.after - self.before
        return None if moved.negative else moved

    @property
    def measured(self) -> bool:
        return self.delta is not None

    @property
    def why(self) -> str:
        """Why there is no measurement, empty when there is one."""
        if self.measured:
            return ""
        if self.note:
            return self.note
        if self.before is None or self.after is None:
            return "the server reports no speculative counters"
        return "the counters went backwards; the server restarted mid-block"

    def public(self) -> dict[str, object]:
        out: dict[str, object] = {"workload": self.workload, "sampling": self.sampling,
                                  "label": self.label, "seconds": round(self.seconds, 3),
                                  "measured": self.measured}
        if (moved := self.delta) is not None:
            out.update(moved.public())
        else:
            out["why"] = self.why
        return out


@contextmanager
def counting(base_url: str, *, workload: str, sampling: str, label: str = "",
             timeout: float = 5.0) -> Iterator[Block]:
    """Read the server's counters either side of the block, as one labelled measurement.

    The counters are the whole server's, so the block must be the only work on it; a
    request still running at either end is recorded as a note rather than silently mixed in.
    """
    block = Block(workload=workload, sampling=sampling, label=label)
    block.before = read_speculative(base_url, timeout=timeout)
    began = monotonic()
    try:
        yield block
    finally:
        block.seconds = monotonic() - began
        block.after = read_speculative(base_url, timeout=timeout)
        busy = [one.processing for one in (block.before, block.after) if one is not None]
        if any(n > 0 for n in busy):
            block.note = "another request was on the server during this block"


@dataclass
class Ledger:
    """Every measured block, per workload and sampling, and pooled over all of them."""

    blocks: list[Block] = field(default_factory=list)

    def add(self, block: Block) -> None:
        self.blocks.append(block)

    def measured(self) -> list[tuple[Block, Speculative]]:
        """Every block that measured, with what it drafted."""
        return [(one, moved) for one in self.blocks
                if (moved := one.delta) is not None]

    def by_workload(self) -> dict[tuple[str, str], Speculative]:
        """One total per ``(workload, sampling)``, over the blocks that measured."""
        out: dict[tuple[str, str], Speculative] = {}
        for one, moved in self.measured():
            key = (one.workload, one.sampling)
            out[key] = _added(out[key], moved) if key in out else moved
        return out

    def overall(self) -> Speculative | None:
        """Every measured block pooled into one, or None where none measured."""
        total: Speculative | None = None
        for _, moved in self.measured():
            total = moved if total is None else _added(total, moved)
        return total

    def lines(self) -> list[str]:
        """The per-workload table and the pooled line, ready to print."""
        head = (f"  {'workload':<10} {'sampling':<10} {'drafts':>8} {'offered':>8} "
                f"{'kept':>8} {'accept':>7} {'depth':>6} {'tok/pass':>8}")
        out = [head, "  " + "-" * (len(head) - 2)]
        for (workload, sampling), moved in sorted(self.by_workload().items()):
            out.append(_row(workload, sampling, moved))
        if (pooled := self.overall()) is not None:
            out.append("  " + "-" * (len(head) - 2))
            out.append(_row("all", "all", pooled))
        for one in self.blocks:
            if not one.measured:
                out.append(f"  {one.workload} at {one.sampling}: not measured -- {one.why}")
        return out

    def public(self) -> dict[str, object]:
        pooled = self.overall()
        return {"blocks": [one.public() for one in self.blocks],
                "by_workload": [{"workload": w, "sampling": s, **moved.public()}
                                for (w, s), moved in sorted(self.by_workload().items())],
                "overall": pooled.public() if pooled is not None else None}


def _added(left: Speculative, right: Speculative) -> Speculative:
    width = max(len(left.per_position), len(right.per_position))
    return Speculative(
        drafts=left.drafts + right.drafts,
        drafted=left.drafted + right.drafted,
        accepted=left.accepted + right.accepted,
        per_position=tuple(_at(left.per_position, i) + _at(right.per_position, i)
                           for i in range(width)),
    )


def _at(values: Sequence[int], index: int) -> int:
    return values[index] if index < len(values) else 0


def _row(workload: str, sampling: str, moved: Speculative) -> str:
    return (f"  {workload:<10} {sampling:<10} {moved.drafts:>8} {moved.drafted:>8} "
            f"{moved.accepted:>8} {_pct(moved.acceptance):>7} "
            f"{_num(moved.depth_offered):>6} {_num(moved.tokens_per_draft):>8}")


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def survival_lines(moved: Speculative, *, most: int = 24) -> list[str]:
    """Acceptance by position within the draft: where the draft stops being worth having."""
    survival = moved.survival[:most]
    if not survival:
        return []
    out = ["  position  survival  marginal"]
    previous = 1.0
    for at, share in enumerate(survival):
        out.append(f"  {at:>8}  {100 * share:>7.1f}%  "
                   f"{100 * (share / previous if previous else 0):>7.1f}%")
        previous = share
    return out


def read_counts(rows: Mapping[str, object]) -> Speculative:
    """A `Speculative` back from what `Speculative.public` wrote."""
    return Speculative(drafts=int(rows.get("drafts") or 0),
                       drafted=int(rows.get("drafted") or 0),
                       accepted=int(rows.get("accepted") or 0),
                       per_position=tuple(int(n) for n in (rows.get("per_position") or ())))
