"""Which model tier this machine can actually hold.

One rule here is easy to get wrong and expensive when it is: **never demote on
ignorance.** If the size of a model is unknown, say so. A missing measurement is not
evidence that a model is too big, and treating it as such silently hands the user a
worse model than their machine could run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ml_stack.contracts.loader import ContractError, load

Profile = Literal["desktop", "mobile"]
Verdict = Literal["fits", "too_big", "unknown"]

_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class Budget:
    """How much of a machine one model may claim."""

    usable_fraction_of_total: float
    overhead_fraction: float
    overhead_floor_bytes: int

    @classmethod
    def for_profile(cls, profile: Profile = "desktop") -> "Budget":
        raw = load("model_tiers.json").get("budget", {})
        if profile not in raw:
            raise ContractError(
                f"model_tiers.json has no budget profile {profile!r} "
                f"(has: {sorted(raw)})"
            )
        entry = raw[profile]
        return cls(
            usable_fraction_of_total=float(entry["usable_fraction_of_total"]),
            overhead_fraction=float(entry["overhead_fraction"]),
            overhead_floor_bytes=int(entry["overhead_floor_bytes"]),
        )

    def available_bytes(self, total_bytes: int) -> int:
        """How many bytes a model is allowed to occupy on a machine this size."""
        return int(total_bytes * self.usable_fraction_of_total)


@dataclass(frozen=True, slots=True)
class Tier:
    """One rung of the ladder."""

    id: str
    min_ram_gb: float
    gguf_repo: str | None
    gguf_file: str | None
    ollama: str | None
    context: int

    @property
    def min_ram_bytes(self) -> int:
        return int(self.min_ram_gb * _GIB)


def _parse(entry: dict[str, Any]) -> Tier:
    gguf = entry.get("gguf") or {}
    return Tier(
        id=str(entry["id"]),
        min_ram_gb=float(entry["min_ram_gb"]),
        gguf_repo=gguf.get("repo"),
        gguf_file=gguf.get("file"),
        ollama=entry.get("ollama"),
        context=int(entry.get("context", 4096)),
    )


def tiers() -> list[Tier]:
    """The ladder, largest first.

    Order is the file's order, not a re-sort: ``tests/test_contracts.py`` asserts the
    file is already descending by ``min_ram_gb``. Re-sorting here would paper over an
    entry inserted in the wrong place instead of failing on it.
    """
    raw = load("model_tiers.json")
    entries = raw.get("tiers")
    if not entries:
        raise ContractError("model_tiers.json has no 'tiers' list")
    return [_parse(e) for e in entries]


def weights_plus_overhead_bytes(weights_bytes: int, budget: Budget) -> int:
    """What a model really costs: weights, plus KV cache / activations / arena.

    The overhead is a fraction with a floor, and it is a rough figure -- describe it as
    one wherever it is shown to a user.
    """
    overhead = max(
        int(weights_bytes * budget.overhead_fraction),
        budget.overhead_floor_bytes,
    )
    return weights_bytes + overhead


def fits(
    weights_bytes: int | None,
    total_bytes: int,
    budget: Budget | None = None,
) -> Verdict:
    """Can this machine hold a model of this size?

    ``weights_bytes is None`` means the size could not be measured -- an unreachable
    model server, a repo that has not been pulled. That returns ``"unknown"``, never
    ``"too_big"``. The caller decides what to do with not knowing; this function refuses
    to turn ignorance into a demotion.
    """
    if weights_bytes is None:
        return "unknown"
    budget = budget or Budget.for_profile("desktop")
    needed = weights_plus_overhead_bytes(weights_bytes, budget)
    return "fits" if needed <= budget.available_bytes(total_bytes) else "too_big"


def largest_that_fits(
    total_bytes: int,
    *,
    available: set[str] | None = None,
    profile: Profile = "desktop",
) -> Tier:
    """Walk the ladder and take the first rung this machine can hold.

    Because the ladder is descending, "the first that fits" and "the largest that fits"
    are the same walk -- which is why the ordering is asserted rather than assumed.

    ``available`` optionally restricts the walk to tiers whose model is actually present
    (e.g. the set of Ollama tags from ``/api/tags``). Passing ``None`` means "do not
    filter", not "nothing is available" -- the same refusal-to-demote-on-ignorance rule
    as ``fits``.
    """
    ladder = tiers()
    for tier in ladder:
        if total_bytes < tier.min_ram_bytes:
            continue
        if available is not None and tier.ollama is not None and tier.ollama not in available:
            continue
        return tier
    return ladder[-1]  # the device rung: always reachable, ships its own weights
