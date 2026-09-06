"""Whether this machine is quiet enough for a timing taken on it to mean anything."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from ml_stack.bench.run import beside_on_the_card
from ml_stack.serve.manager import measurement_on_the_card, measurement_said
from ml_stack.units import human_bytes

__all__ = ["Quiet", "look"]

#: Load average per core above which the machine is doing something else.
BUSY_LOAD = 0.5
#: CPU busy percent above which it is. Four agent processes at 190% each on sixteen cores
#: is 0.47 per core and 47% busy: under the load bar and well over this one.
BUSY_PERCENT = 35.0


def _load_per_core() -> float:
    try:
        return os.getloadavg()[0] / max(1, os.cpu_count() or 1)
    except (OSError, AttributeError):
        return 0.0


def _busy_percent() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return float(psutil.cpu_percent(interval=0.5))
    except (OSError, ValueError, RuntimeError):
        return None


@dataclass(frozen=True)
class Quiet:
    """What else is on this machine, and whether a timing taken now can be trusted."""

    servers: tuple[dict[str, Any], ...] = ()
    measuring: dict[str, Any] | None = None
    load: float = 0.0
    busy: float | None = None
    cores: int = 1
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Whether nothing found would put itself into a timing."""
        return not self.reasons

    def lines(self) -> list[str]:
        """What was found, one line each, whether or not any of it is a problem."""
        out = []
        for one in self.servers:
            out.append(f"  :{one['port']} {one['model'] or '?'} "
                       f"{human_bytes(one['bytes'])} resident, pid {one['pid']}"
                       + ("" if one["leased"] else ", not leased by ml-stack"))
        if not self.servers:
            out.append("  no model server is running")
        out.append(f"  measuring lock: {measurement_said(self.measuring)}"
                   if self.measuring else "  measuring lock: free")
        said = f"  load {self.load:.2f} per core over {self.cores} core(s)"
        if self.busy is not None:
            said += f", cpu {self.busy:.0f}% busy"
        out.append(said)
        return out

    def refusal(self) -> list[str]:
        """What to stop before measuring, in words a person can act on."""
        out = ["this machine is not quiet, so a timing taken on it would not be this "
               "model's:"]
        out += [f"  {why}" for why in self.reasons]
        out.append("stop those and run this again, or pass --anyway to measure regardless "
                   "-- every row is then marked as taken on a busy machine.")
        return out


def look() -> Quiet:
    """What else is on this machine right now, and why a timing would not be trusted."""
    servers = tuple(beside_on_the_card())
    held = measurement_on_the_card()
    load, busy = _load_per_core(), _busy_percent()
    cores = max(1, os.cpu_count() or 1)

    reasons: list[str] = []
    for one in servers:
        reasons.append(
            f"a model server holds this card on port {one['port']} "
            f"({one['model'] or 'unknown model'}, {human_bytes(one['bytes'])} resident, "
            f"pid {one['pid']}) -- 'ml-stack-serve down --port {one['port']}' stops it. "
            f"A second model merely holding memory costs throughput.")
    if held is not None:
        reasons.append(f"a measurement holds the card: {measurement_said(held)} -- wait "
                       f"for it, or stop it with 'ml-stack-bench stop'.")
    if load > BUSY_LOAD or (busy is not None and busy > BUSY_PERCENT):
        reasons.append(f"the machine is busy: load {load:.2f} per core over {cores} "
                       f"core(s)"
                       + (f", cpu {busy:.0f}% busy" if busy is not None else "")
                       + f" -- the bars are {BUSY_LOAD:g} per core and {BUSY_PERCENT:g}%. "
                       f"Find what is running and stop it; seconds measured beside other "
                       f"work are not this model's.")
    return Quiet(servers=servers, measuring=held, load=load, busy=busy, cores=cores,
                 reasons=tuple(reasons))
