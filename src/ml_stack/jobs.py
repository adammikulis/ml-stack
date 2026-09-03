"""A record for a command that outlives the terminal that started it.

The ingest keeps one of these by hand (``ingesting.json``, ``stop``, ``wait``); the bench
keeps another (``measuring.json``, ``status``, ``tail``, ``stop``); training has none. Each
reinvented the same three facts -- the pid, where its log is, and whether it is still alive
-- in its own file. This module is that record, once: one JSON file per *kind* of long
command, so ``wait`` and ``stop`` never need a hand-written ``pgrep`` loop and a command that
chains after another is always ``wait && next``.
"""

from __future__ import annotations

import json
import os
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.lock import Busy

HOME = Path(os.environ.get("MLSTACK_JOBS_HOME") or "~/.ml-stack/jobs").expanduser()
"""Where a record lives when a caller names no ``home`` of its own: ``HOME/<kind>.json``."""

STOP_WAIT = 60.0
"""How long `stop` waits for the pid to end before saying it is still going."""

__all__ = ["Job", "record", "alive", "wait", "stop", "status", "HOME", "STOP_WAIT"]


@dataclass(frozen=True, slots=True)
class Job:
    """One long command's record: what it was, its pid, argv, log, when it started."""

    kind: str
    pid: int
    argv: tuple[str, ...] = field(default_factory=tuple)
    log: str = ""
    started: str = ""
    home: Path = HOME

    def as_dict(self) -> dict[str, Any]:
        return {"pid": self.pid, "argv": list(self.argv), "log": self.log,
                "started": self.started}


def _path(kind: str, home: Path | None) -> Path:
    return (home or HOME) / f"{kind}.json"


def _read(path: Path) -> dict[str, Any]:
    try:
        held = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return held if isinstance(held, dict) else {}


def _ended(pid: int) -> bool:
    """Whether ``pid`` is gone. A process this one started is reaped on the way: a child
    that has exited answers ``kill(pid, 0)`` until somebody collects it."""
    try:
        collected, _ = os.waitpid(pid, os.WNOHANG)
    except (OSError, AttributeError, ValueError):
        collected = 0
    if collected:
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    return False


def _wait_for(pid: int, seconds: float) -> bool:
    deadline = time.time() + max(seconds, 0.0)
    while not _ended(pid):
        if time.time() >= deadline:
            return False
        time.sleep(0.05)
    return True


def alive(kind: str, *, home: Path | None = None) -> int:
    """The pid recorded for ``kind`` while it is still running; 0 when there is none, or the
    recorded pid has ended."""
    held = _read(_path(kind, home))
    pid = int(held.get("pid") or 0)
    if not pid:
        return 0
    return 0 if _ended(pid) else pid


def record(kind: str, *, pid: int, argv: Sequence[str] = (), log: str = "", started: str = "",
          home: Path | None = None, refuse_if_alive: bool = True) -> Job:
    """Write ``kind``'s record for a command already started (``pid``): what it was called
    with and where its log is.

    Refuses with `Busy` when a previous ``kind`` is still alive -- so a caller checks
    `alive` before spawning a second one and never overwrites a run still running. A caller
    whose own concurrency is handled elsewhere (the bench queues a second `--detach` behind
    a file lock rather than refusing it) passes ``refuse_if_alive=False``.
    """
    home = home or HOME
    if refuse_if_alive:
        held = alive(kind, home=home)
        if held:
            raise Busy(f"a {kind} job (pid {held}) is already running")
    home.mkdir(parents=True, exist_ok=True)
    job = Job(kind=kind, pid=int(pid), argv=tuple(str(a) for a in argv), log=str(log),
              started=started or time.strftime("%FT%T"), home=home)
    _path(kind, home).write_text(json.dumps(job.as_dict(), indent=1), encoding="utf-8")
    return job


def wait(kind: str, *, say: Callable[[str], None] = print, every: float = 60.0,
         home: Path | None = None) -> int:
    """Block until the ``kind`` job this machine records has ended, saying so every
    ``every`` seconds -- so the next command can follow it without a loop written by hand
    (``ml-stack-bench wait && ml-stack-bench report --profile``)."""
    pid = alive(kind, home=home)
    if not pid:
        say(f"no {kind} job is running")
        return 0
    waited = 0.0
    step = min(every, 5.0) or 5.0
    while alive(kind, home=home):
        time.sleep(step)
        waited += step
        if waited % every < step:
            say(f"  still running (pid {pid}) after {waited / 60:.0f} min")
    say(f"the {kind} job (pid {pid}) has ended")
    return 0


def stop(kind: str, *, say: Callable[[str], None] = print, wait: float = STOP_WAIT,
         home: Path | None = None) -> int:
    """``SIGTERM`` to the recorded ``kind`` pid, and wait up to ``wait`` seconds for it to
    end, saying so every 30s while it has not. The record is kept while it is still ending
    -- so a caller checking `alive` first never starts a second one beside it -- and removed
    once it has, or once the pid was already gone."""
    home = home or HOME
    path = _path(kind, home)
    held = _read(path)
    pid = int(held.get("pid") or 0)
    if not pid:
        say(f"no {kind} job is recorded")
        return 1
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        say(f"the recorded {kind} job (pid {pid}) had already ended")
        path.unlink(missing_ok=True)
        return 1
    waited = 0.0
    ended = _wait_for(pid, min(wait, 30.0))
    waited += min(wait, 30.0)
    while not ended and waited < wait:
        say(f"  still ending after {waited:.0f}s (pid {pid})")
        ended = _wait_for(pid, min(30.0, wait - waited))
        waited += 30.0
    if not ended:
        say(f"asked the {kind} job (pid {pid}) to stop; it had not ended after {wait:.0f}s -- "
            f"its record stays, and `alive` still says it is running")
        return 1
    path.unlink(missing_ok=True)
    say(f"stopped the {kind} job (pid {pid})")
    return 0


def status(*, say: Callable[[str], None] = print, home: Path | None = None) -> int:
    """Every kind's record under ``home``: running or ended, since when, its log."""
    home = home or HOME
    records = sorted(home.glob("*.json")) if home.exists() else []
    if not records:
        say("no job is recorded")
        return 0
    for path in records:
        kind = path.stem
        held = _read(path)
        pid = int(held.get("pid") or 0)
        running = bool(pid) and not _ended(pid)
        state = f"running (pid {pid})" if running else (f"ended (pid {pid})" if pid else "unknown")
        say(f"{kind}: {state} since {held.get('started', '?')}")
        argv = " ".join(str(a) for a in (held.get("argv") or ()))
        if argv:
            say(f"  argv: {argv}")
        if held.get("log"):
            say(f"  log: {held['log']}")
    return 0
