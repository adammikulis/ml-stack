"""Coalesce runtime builds: one in flight, only the newest tip, and only after the tip has stopped moving."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ml_stack import runtime_deploy

STABLE_S = 300.0
POLL_S = 15.0
MAX_ROUNDS = 8
LOG_CAP = 262144

Retarget = Callable[[], runtime_deploy.Plan]


@dataclass(frozen=True, slots=True)
class Pace:
    """How long the tip must stand still, how often it is looked at, and the clock and sleep that measure it."""

    stable_s: float = STABLE_S
    poll_s: float = POLL_S
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


PACE = Pace()


def settle(retarget: Retarget, pace: Pace = PACE) -> runtime_deploy.Plan:
    """The plan for the newest tip once it has been unchanged for `pace.stable_s` seconds; a burst of merges yields one plan."""
    plan, since = retarget(), pace.clock()
    while pace.clock() - since < pace.stable_s:
        pace.sleep(min(pace.poll_s, max(pace.stable_s - (pace.clock() - since), 0.0)))
        newer = retarget()
        if newer.commit != plan.commit:
            plan, since = newer, pace.clock()
    return plan


def coalesced(retarget: Retarget, run: Callable[[runtime_deploy.Plan], runtime_deploy.Outcome],
              pace: Pace = PACE) -> runtime_deploy.Outcome:
    """Settle, run, and when the tip moved during the run settle and run again; commits in between are never built."""
    outcome = runtime_deploy.Outcome("current")
    for _ in range(MAX_ROUNDS):
        plan = settle(retarget, pace)
        outcome = run(plan)
        if retarget().commit == plan.commit or not outcome.ok:
            break
    return outcome


def rotate(log: Path, cap: int = LOG_CAP) -> bool:
    """Empty a log past `cap` bytes in place; nothing is written, and appenders carry on."""
    try:
        if log.stat().st_size <= cap:
            return False
    except OSError:
        return False
    os.truncate(log, 0)
    return True
