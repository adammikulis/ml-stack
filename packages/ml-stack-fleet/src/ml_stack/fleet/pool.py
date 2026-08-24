"""Choosing which peer runs a piece of work, and saying why when none can.

``Peer.find_one`` refuses to pick between two matching peers on purpose: interactively,
"several match" means the human has not said what they meant. This module is the other
entry point, where "several" is the input rather than the problem. It does not relax
that rule; it sits beside it.

Three things shape everything here.

**Eligibility is a separate step from scoring.** A ``-inf`` score used as a filter throws
away the reason, and then "nothing ran" is a hang with no explanation. ``eligible``
returns what was kept *and* why each of the rest was dropped, so an unplaceable unit
fails immediately with the whole picture.

**Speed is measured, never assumed.** There is no table saying a 4090 beats a Pi; it
would be wrong within a year, and it is the wrong instinct in a codebase whose contracts
package refuses to demote a model on ignorance. Rates come from jobs that actually ran.

**An unmeasured peer is not a slow peer.** A missing measurement is not evidence -- the
same rule ``fits()`` follows. An unmeasured peer scores at the median of the measured
ones and is given an explore allowance, so a box that has never run anything still gets
work while a known-fast box is busy. Score it as slow instead and the first peer ever
measured wins forever, because nothing else is ever tried.
"""

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
    """What a piece of work needs from a machine, and what it must not take.

    Measured constraints (``backend``, ``min_vram_gb``) and declared ones (``labels``)
    compose deliberately: a box can *prove* it has CUDA, but it cannot prove it is meant
    for data prep, so that half has to be declared with ``traind --label``.
    """

    backend: str = ""
    labels: tuple[str, ...] = ()
    exclude_labels: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    min_vram_gb: float = 0.0
    min_ram_gb: float = 0.0
    min_cpus: int = 0
    exclusive: bool = False
    """Take the whole box. Advisory: it refuses a peer that is not entirely free, but two
    coordinators can still collide -- ``--slots`` is the only real enforcement."""

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
                return (f"does not report {self.backend!r}; "
                        f"has {sorted(backends) or 'no backends'}")

        for key, need, unit in (("vram_free_gb", self.min_vram_gb, "GB VRAM"),
                                ("ram_gb", self.min_ram_gb, "GB RAM"),
                                ("cpus", self.min_cpus, "CPUs")):
            if not need:
                continue
            have = report.get(key)
            if have is None:
                # Unmeasured is not "too small". Refusing here would exclude every box
                # running the stdlib-only probe, which is most of them.
                continue
            if have < need:
                return f"has {have} {unit}, needs {need}"

        if self.exclusive and free < slots:
            return f"is not idle ({free}/{slots} free) and this work wants the whole box"
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
    """Measured units/second for this kind of work. ``None`` means never measured, which
    is not the same as slow."""
    bench: float | None = None
    """This peer's join-time benchmark score, if it has one. A proxy for real work and
    never a substitute for ``rate`` -- but it is a *measurement*, so it orders peers that
    have not yet run this kind of work far better than picking among them arbitrarily."""
    transfer_gb: float = 0.0
    """Bytes this unit would have to fetch to run here, in GB."""


def candidates(peers: Sequence[Peer], *, kind: str = "",
               rates: Mapping[tuple[str, str], float] | None = None,
               bench: Mapping[str, float] | None = None,
               held: Callable[[Peer], float] | None = None) -> list[Candidate]:
    """Ask every peer what it looks like now. Unreachable peers drop out silently.

    Deliberately live rather than reading the beacon: a beacon is up to one announcement
    interval stale, and the gap between "was idle when it last announced" and "is idle"
    is exactly the window two coordinators race in.
    """
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
    """Score by estimated seconds until this unit is *finished* here. Lower is better.

    Transfer is part of the estimate rather than a tiebreak: a slower box that already
    holds the shard usually beats a faster one that has to fetch two gigabytes first,
    and a score that ignores that will keep choosing the fast idle box and keep waiting
    on the network.

    ``typical`` is what an unmeasured peer is scored at, and the caller supplies it --
    the median of the peers that *have* been measured. Scoring an unmeasured peer as
    slow would mean the first peer ever measured wins forever, because nothing else
    would ever be tried.
    """
    def score(c: Candidate, typical: float) -> float:
        rate = c.rate or typical
        run_s = work / rate if rate > 0 else work
        transfer_s = c.transfer_gb * 8 / link_gbps if c.transfer_gb else 0.0
        wait_s = max(0, c.queued - c.free) * queue_penalty_s
        return run_s + transfer_s + wait_s

    return score


def choose(cands: Sequence[Candidate], *, score: "Score | None" = None,
           explore: bool = True) -> Candidate | None:
    """The peer to use, or ``None`` when every one of them is full.

    With ``explore``, an unmeasured peer holding a free slot wins outright. That is the
    "a missing measurement is not evidence" rule made operational: the only way a peer
    stops being unmeasured is by being given something to do.
    """
    free = [c for c in cands if c.free > 0]
    if not free:
        return None
    if explore:
        unmeasured = [c for c in free if c.rate is None]
        if unmeasured:
            # Among peers with nothing measured for THIS kind of work, prefer the one
            # the join-time benchmark liked most, then the one with the most capacity.
            # An unbenchmarked peer still sorts ahead of a slow benchmarked one, because
            # "not measured" must not quietly become "measured badly".
            return max(unmeasured,
                       key=lambda c: (c.bench if c.bench is not None else float("inf"),
                                      c.free))

    measured = [c.rate for c in free if c.rate]
    typical = median(measured) if measured else 1.0
    fn = score or soonest()
    return min(free, key=lambda c: fn(c, typical))
