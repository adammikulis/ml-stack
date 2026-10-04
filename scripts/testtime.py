"""The full tier's wall and CPU time as a ratchet: it only falls.

``tests/full-tier-time.json`` records the last accepted time of the full tier and the number of
workers it had. A full run writes what it measured to ``.full-tier-last.json`` (not committed) and
``scripts/test ratchet`` compares the two:

* CPU seconds (the sum of every test's own time) are compared always: how much work the suite does
  hardly depends on the machine being busy.
* Wall seconds are compared only when the run had at least as many workers as the record, since a
  run on fewer workers is slower for a reason that is not the tests.
* More than ``TOLERANCE`` over the record fails. ``--update`` writes a lower record and refuses a
  higher one (an agent can never raise it; the owner may with ``--allow-increase`` at a terminal).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

TOLERANCE = 0.10
RECORD = "tests/full-tier-time.json"
LAST = ".full-tier-last.json"
CASE = re.compile(r'<testcase [^>]*?time="([^"]+)"')


@dataclass(frozen=True)
class Timing:
    """One full-tier run: wall seconds, summed test seconds, workers and tests."""

    wall_s: float
    cpu_s: float
    workers: int
    tests: int
    load: float = 0.0
    """The machine's one-minute load average when the run began; only kept in the last-run file."""


def cpu_seconds(junit_text: str) -> tuple[float, int]:
    """The summed time and the count of the testcases in a junit file."""
    times = [float(t) for t in CASE.findall(junit_text)]
    return sum(times), len(times)


def load(path: Path) -> Timing | None:
    """A saved timing, or None when there is none or it is unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Timing(float(data["wall_s"]), float(data["cpu_s"]), int(data["workers"]),
                      int(data.get("tests", 0)), float(data.get("load", 0.0)))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save(path: Path, timing: Timing) -> None:
    """Write a timing as sorted, rounded JSON."""
    body = {"load": round(timing.load, 1)} if timing.load else {}
    body |= {"cpu_s": round(timing.cpu_s), "tests": timing.tests, "wall_s": round(timing.wall_s),
            "workers": timing.workers}
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def busy(now: Timing) -> bool:
    """Whether the machine was too loaded for the wall time to say anything about the tests."""
    return now.load > (os.cpu_count() or 4) / 2


def verdict(recorded: Timing, now: Timing) -> list[str]:
    """What is over the record, one line each; empty when the run is within tolerance."""
    over = []
    if now.cpu_s > recorded.cpu_s * (1 + TOLERANCE):
        over.append(f"CPU time {now.cpu_s:.0f} s is more than {TOLERANCE:.0%} over the recorded "
                    f"{recorded.cpu_s:.0f} s")
    if not busy(now) and now.workers >= recorded.workers and now.wall_s > recorded.wall_s * (1 + TOLERANCE):
        over.append(f"wall time {now.wall_s:.0f} s on {now.workers} workers is more than "
                    f"{TOLERANCE:.0%} over the recorded {recorded.wall_s:.0f} s on "
                    f"{recorded.workers}")
    return over


def lowered(recorded: Timing | None, now: Timing) -> Timing:
    """The record after an update: each number may fall, never rise (wall is only comparable at
    the same or more workers, otherwise the old one is kept)."""
    if recorded is None:
        return now
    wall, workers = recorded.wall_s, recorded.workers
    if not busy(now) and now.workers >= recorded.workers and now.wall_s < recorded.wall_s:
        wall, workers = now.wall_s, now.workers
    return Timing(wall, min(recorded.cpu_s, now.cpu_s), workers, now.tests)


def check(root: Path, update: bool = False, allow_increase: bool = False) -> int:
    """Compare the last full run with the record; 0 when within tolerance."""
    now = load(root / LAST)
    if now is None:
        print("ratchet: no full-tier run recorded here yet (scripts/test full writes one)")
        return 2
    recorded = load(root / RECORD)
    if update:
        if recorded is not None and verdict(recorded, now) and not allow_increase:
            print("ratchet: refusing to raise the record:\n  " + "\n  ".join(verdict(recorded, now)))
            return 1
        new = now if allow_increase or recorded is None else lowered(recorded, now)
        save(root / RECORD, new)
        print(f"ratchet: recorded wall {new.wall_s:.0f} s on {new.workers} workers, "
              f"CPU {new.cpu_s:.0f} s")
        return 0
    if recorded is None:
        print("ratchet: nothing recorded; scripts/test ratchet --update records this run")
        return 1
    over = verdict(recorded, now)
    print(f"ratchet: now wall {now.wall_s:.0f} s ({now.workers} workers), CPU {now.cpu_s:.0f} s; "
          f"recorded wall {recorded.wall_s:.0f} s ({recorded.workers}), CPU {recorded.cpu_s:.0f} s")
    if busy(now):
        print(f"ratchet: wall time not compared: the load was {now.load:.0f} when the run began")
    for line in over:
        print(f"ratchet: {line}")
    return 1 if over else 0
