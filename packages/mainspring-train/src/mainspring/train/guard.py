"""Things that stop a long run from wasting itself.

Three separate hazards, each with its own guard:

* **Two runs in one output directory.** Both write checkpoints, each overwrites the
  other's, and the resulting run resumes from an interleaving of two different models.
* **A run that has gone numerically bad.** One NaN batch is a hardware blip; a thousand is
  a broken run that has been burning a GPU for hours producing nothing.
* **A run that has stalled.** Throughput collapses -- memory pressure, thermal throttling,
  a swap storm -- and the run keeps going at a fraction of its speed with no error.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType


class RunLockError(RuntimeError):
    """Another process already owns this output directory."""


class TrainingDiverged(RuntimeError):
    """Too many non-finite steps. The run is over."""


class RunLock:
    """Exclusive ownership of an output directory, for the life of the process.

    ``flock`` rather than a pid file, because the kernel releases it when the process dies.
    A pid file left behind by a crash blocks every subsequent run until a human deletes it,
    which trains people to delete it reflexively -- and then it stops protecting anything.
    """

    def __init__(self, directory: Path | str, *, name: str = "run.lock") -> None:
        self.path = Path(directory) / name
        self._handle = None

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def acquire(self) -> None:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RunLockError(
                f"another process is already training into {self.path.parent}. "
                "Two runs sharing an output directory overwrite each other's checkpoints."
            ) from exc
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        import fcntl

        try:
            fcntl.flock(self._handle, fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
            self.path.unlink(missing_ok=True)


@dataclass
class NonFiniteBudget:
    """How many non-finite steps to skip before giving up.

    Skipping is right: a single bad step is usually a transient hardware fault, and
    aborting the run over one wastes everything before it. Skipping *without a budget* is
    wrong: it converts a permanently broken GPU into a run that trains on nothing for as
    long as you let it.

    The budget is cumulative over the whole run, not consecutive. A machine producing one
    bad step in every ten is broken even though it never produces two in a row.
    """

    max_skipped: int = 50
    skipped: int = 0
    last_step: int | None = None

    def record_skip(self, step: int) -> None:
        self.skipped += 1
        self.last_step = step
        if self.skipped > self.max_skipped:
            raise TrainingDiverged(
                f"skipped {self.skipped} non-finite steps (limit {self.max_skipped}); "
                f"most recent at step {step}. This is a broken run, not a blip -- check "
                "the hardware before restarting."
            )

    @property
    def exhausted(self) -> bool:
        return self.skipped > self.max_skipped


@dataclass
class StallWatchdog:
    """Notice when steps suddenly get much slower.

    Compares each step against the rolling **median**, not the mean: one 40-second stall
    drags a mean far enough that the next stall looks normal, which is exactly backwards.

    This reports; it does not intervene. What to do about a stall depends on the run.
    """

    window: int = 101
    factor: float = 3.0
    absolute_s: float = 30.0
    durations: list[float] = field(default_factory=list)

    def record(self, duration_s: float) -> str | None:
        """Record a step duration. Returns a message if it looks like a stall."""
        self.durations.append(duration_s)
        if len(self.durations) > self.window:
            self.durations.pop(0)
        if len(self.durations) < 10:
            return None  # not enough history for a median to mean anything

        median = statistics.median(self.durations)
        threshold = max(self.factor * median, median + self.absolute_s)
        if duration_s <= threshold:
            return None
        return (
            f"step took {duration_s:.1f}s against a median of {median:.1f}s "
            f"(threshold {threshold:.1f}s) -- likely memory pressure, thermal "
            "throttling, or swap"
        )

    @property
    def median_s(self) -> float:
        return statistics.median(self.durations) if self.durations else 0.0


class StepTimer:
    """Context manager timing one step. ``with timer: ...`` then read ``timer.elapsed``."""

    def __init__(self) -> None:
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> "StepTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed = time.perf_counter() - self._start
