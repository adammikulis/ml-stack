"""Which model tier this machine can actually hold."""

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
    """The ladder, largest first."""
    raw = load("model_tiers.json")
    entries = raw.get("tiers")
    if not entries:
        raise ContractError("model_tiers.json has no 'tiers' list")
    return [_parse(e) for e in entries]


def weights_plus_overhead_bytes(weights_bytes: int, budget: Budget) -> int:
    """What a model really costs: weights, plus KV cache / activations / arena."""
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
    """Can this machine hold a model of this size?"""
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
    """Walk the ladder and take the first rung this machine can hold."""
    ladder = tiers()
    for tier in ladder:
        if total_bytes < tier.min_ram_bytes:
            continue
        if available is not None and tier.ollama is not None and tier.ollama not in available:
            continue
        return tier
    return ladder[-1]  # the device rung: always reachable, ships its own weights
