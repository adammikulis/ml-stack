"""Choosing which peer runs a piece of work, and saying why when none can."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from .remote import Peer

__all__ = ["Candidate", "Requires", "Score", "candidates", "choose", "eligible",
           "soonest"]

Score = Callable[["Candidate", float], float]
"""Seconds until done, given the candidate and the median rate of the measured peers."""


@dataclass(frozen=True, slots=True)
class Requires:
    """What a piece of work needs from a machine, and what it must not take."""

    backend: str = ""
    labels: tuple[str, ...] = ()
    exclude_labels: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    min_vram_gb: float = 0.0
    min_ram_gb: float = 0.0
    min_cpus: int = 0
    exclusive: bool = False
    """Take the whole box. Advisory: it refuses a peer that is not entirely free, but two"""

    def why_not(self, peer_name: str, report: Mapping[str, Any],
                free: int, slots: int) -> str:
        """Empty when this peer admits the work, otherwise the reason it does not."""
        if self.names and peer_name not in self.names:
            return f"name {peer_name!r} is not one of {list(self.names)}"

        labels = set(report.get("labels") or ())
        missing = [lbl for lbl in self.labels if lbl not in labels]
        if missing:
            return f"missing label(s) {missing}; declares {sorted(labels) or 'none'}"
        clashing = [lbl for lbl in self.exclude_labels if lbl in labels]
        if clashing:
            return f"declares excluded label(s) {clashing}"

        if self.backend:
            backends = report.get("backends") or []
            if not (report.get(self.backend) or self.backend in backends):
                have = sorted(backends) or "no backends"
                extra = ""
                if self.backend == "cuda" and report.get("rocm"):
                    extra = " (it has ROCm -- ask for 'rocm', or 'accelerator' for any GPU)"
                elif self.backend in ("cuda", "rocm") and report.get("accelerator"):
                    extra = f" (it has {report.get('vendor', 'some')} acceleration)"
                return f"does not report {self.backend!r}; has {have}{extra}"

        for key, need, unit in (("vram_free_gb", self.min_vram_gb, "GB VRAM"),
                                ("ram_gb", self.min_ram_gb, "GB RAM"),
                                ("cpus", self.min_cpus, "CPUs")):
            if not need:
                continue
            have = report.get(key)
            if have is None:
                continue
            if have < need:
                return f"has {have} {unit}, needs {need}"

        if self.exclusive and free < slots:
            return f"is not idle ({free}/{slots} free) and this work wants the whole box"

        schedule = report.get("availability") or {}
        if schedule and not schedule.get("available", True):
            return schedule.get("unavailable_because") or "is not taking work right now"
        return ""

    def admits(self, peer_name: str, report: Mapping[str, Any],
               free: int, slots: int) -> bool:
        return not self.why_not(peer_name, report, free, slots)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One peer, as it looks right now, for one piece of work."""

    peer: Peer
    name: str
    report: Mapping[str, Any] = field(default_factory=dict)
    slots: int = 1
    free: int = 0
    queued: int = 0
    rate: float | None = None
    """Measured units/second for this kind of work. ``None`` means never measured, which"""
    bench: float | None = None
    """This peer's join-time benchmark score, if it has one. A proxy for real work and"""
    transfer_gb: float = 0.0
    """Bytes this unit would have to fetch to run here, in GB."""


def candidates(peers: Sequence[Peer], *, kind: str = "",
               rates: Mapping[tuple[str, str], float] | None = None,
               bench: Mapping[str, float] | None = None,
               held: Callable[[Peer], float] | None = None) -> list[Candidate]:
    """Ask every peer what it looks like now. Unreachable peers drop out silently."""
    out: list[Candidate] = []
    for peer in peers:
        try:
            health = peer.health()
        except Exception:                             # noqa: BLE001
            continue
        name = health.get("name") or peer.name
        slots = int(health.get("slots") or 1)
        out.append(Candidate(
            peer=peer, name=name, report=health, slots=slots,
            free=int(health.get("free", slots if not health.get("busy") else 0)),
            queued=int(health.get("queued") or 0),
            rate=(rates or {}).get((name, kind)),
            bench=(bench or {}).get(name),
            transfer_gb=held(peer) if held else 0.0,
        ))
    return out


def eligible(cands: Sequence[Candidate],
             requires: Requires) -> tuple[list[Candidate], dict[str, str]]:
    """Split into peers that admit the work and peers that do not, with reasons."""
    kept, refused = [], {}
    for c in cands:
        reason = requires.why_not(c.name, c.report, c.free, c.slots)
        if reason:
            refused[c.name] = reason
        else:
            kept.append(c)
    return kept, refused


def soonest(work: float = 1.0, *, link_gbps: float = 1.0,
            queue_penalty_s: float = 60.0) -> "Score":
    """Score by estimated seconds until this unit is *finished* here. Lower is better."""
    def score(c: Candidate, typical: float) -> float:
        rate = c.rate or typical
        run_s = work / rate if rate > 0 else work
        transfer_s = c.transfer_gb * 8 / link_gbps if c.transfer_gb else 0.0
        wait_s = max(0, c.queued - c.free) * queue_penalty_s
        return run_s + transfer_s + wait_s

    return score


def choose(cands: Sequence[Candidate], *, score: "Score | None" = None,
           explore: bool = True) -> Candidate | None:
    """The peer to use, or ``None`` when every one of them is full."""
    free = [c for c in cands if c.free > 0]
    if not free:
        return None
    if explore:
        unmeasured = [c for c in free if c.rate is None]
        if unmeasured:
            return max(unmeasured,
                       key=lambda c: (c.bench if c.bench is not None else float("inf"),
                                      c.free))

    measured = [c.rate for c in free if c.rate]
    typical = median(measured) if measured else 1.0
    fn = score or soonest()
    return min(free, key=lambda c: fn(c, typical))
