"""Run records of the test broker (queued, running) and the queue forecast built on them."""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import testslots_policy as policy

from ml_stack.lock import release, take

QUEUED, RUNNING = "queued", "running"


class RunRecord:
    """A run's record file, locked while the process lives: class, estimate, workers wanted and granted, times."""

    def __init__(self, directory: Path, label: str, run_class: str, estimate_s: float | None, want: int):
        self.path = directory / f"{time.time_ns()}-{os.getpid()}.run"
        self.stream = self.path.open("w+")
        take(self.stream)
        self.data = {"id": self.path.stem, "pid": os.getpid(), "label": label, "class": run_class, "state": QUEUED,
                     "estimate_s": estimate_s, "want": want, "granted": 0, "enqueued": time.time(),
                     "started": None}
        self._write()

    def _write(self) -> None:
        self.stream.seek(0)
        self.stream.truncate()
        json.dump(self.data, self.stream)
        self.stream.flush()

    def update(self, **changes) -> None:
        """Change fields of the record (state, granted, started)."""
        self.data.update(changes)
        self._write()

    def close(self) -> None:
        """Remove the record."""
        self.path.unlink(missing_ok=True)
        self.stream.close()


def read_runs(directory: Path) -> list[dict]:
    """The records of runs whose process is alive."""
    out = []
    for path in sorted(directory.glob("*.run")):
        with contextlib.suppress(OSError, ValueError):
            fd = os.open(path, os.O_RDWR)
            try:
                if take(fd):
                    release(fd)
                    path.unlink(missing_ok=True)
                    continue
            finally:
                os.close(fd)
            out.append(json.loads(path.read_text()))
    return out


@dataclass
class Forecast:
    """Where a request stands: position among queued requests, work ahead of it, expected start."""
    state: str
    position: int | None
    ahead_count: int
    ahead_estimate_s: float
    start_at: float | None


def _remaining(run: dict, now: float) -> float:
    return max(0.0, float(run.get("estimate_s") or 0.0) - (now - float(run.get("started") or now)))


def _workers(run: dict, budget: int) -> int:
    return max(1, min(int(run.get("want") or budget), budget))


def forecast(runs: list[dict], budget: int, now: float) -> dict[str, Forecast]:
    """The Forecast of every run by record id, under the real ordering (aged first, then least estimated work less wait).

    A queued run starts when a worker is free in a simulation of the running runs' remaining estimated time
    followed by the queued runs ahead of it.
    """
    running = [r for r in runs if r["state"] == RUNNING]
    queued = sorted((r for r in runs if r["state"] == QUEUED),
                    key=lambda r: (policy.sort_key({"class": r["class"], "since": r["enqueued"], "estimate": r["estimate_s"]}, now),
                                       r["enqueued"]))
    live = sorted((now + _remaining(r, now), int(r["granted"])) for r in running)
    result = {r["id"]: Forecast(RUNNING, None, 0, 0.0, r.get("started")) for r in running}
    ahead_s = sum(_remaining(r, now) for r in running)
    clock = now
    for index, run in enumerate(queued):
        while live and budget - sum(w for _, w in live) < 1:
            end, _ = live[0]
            clock = max(clock, end)
            live = live[1:]
        free = budget - sum(w for _, w in live)
        result[run["id"]] = Forecast(QUEUED, index + 1, index, ahead_s, clock)
        workers = min(_workers(run, budget), max(1, free))
        size = float(run.get("estimate_s") or 0.0)
        live = sorted([*live, (clock + size / workers, workers)])
        ahead_s += size
    return result


def describe(run: dict, view: Forecast) -> str:
    """One status line for a run record."""
    est = run.get("estimate_s")
    size = f"est {est:.0f}s" if est is not None else "est ?"
    def stamp(t: float) -> str:
        return dt.datetime.fromtimestamp(t).strftime("%a %H:%M")

    if view.state == QUEUED:
        where = (f"position {view.position}, {view.ahead_count} queued / {view.ahead_estimate_s:.0f}s ahead, "
                 f"starts about {stamp(view.start_at)}")
    else:
        where = f"started {stamp(view.start_at)} on {run['granted']} worker(s)"
    return f"  run {view.state:<8} {policy.run_class(run):<11} pid {run['pid']:<7} {size}; {where}; {run['label']}"
