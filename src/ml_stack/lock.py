"""One runner at a time, without anybody hand-writing a waiter.

Two measurements sharing a GPU produce timings that belong to neither, so a second run has
to wait for the first. The obvious way to arrange that by hand is a shell loop:

    until ! pgrep -f "python -m ml_stack.graph.bench"; do sleep 20; done; <the real command>

which cannot work, and fails silently rather than loudly: the waiting shell's own command
line contains the pattern, so `pgrep -f` matches the waiter itself, the condition is
permanently true, and the loop spins for the life of the machine. Two of them sat there for
an afternoon and the work queued behind them -- a draft-head comparison -- simply never ran.
Nothing announced it. The log file was never created.

So the waiting belongs to the program that knows it must not overlap, not to whoever is
calling it. `only_one` takes an exclusive lock on a file; a second caller either waits for
it or is refused, and either way says so.
"""

from __future__ import annotations

import errno
import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class Busy(RuntimeError):
    """Somebody else holds the lock and we were told not to wait."""


@contextmanager
def only_one(what: str | Path, *, wait: bool = True, timeout: float = 0.0,
             announce=print) -> Iterator[Path]:
    """Hold `what` exclusively for the block, so two of these never run at once.

    `wait=False` raises `Busy` instead of waiting. `timeout` bounds the wait in seconds; 0
    waits as long as it takes. The holder's pid is written into the file so a person looking
    at a stalled machine can see who has it, and it is announced rather than waited on
    silently -- a wait nobody can see is indistinguishable from a hang.
    """
    path = Path(what).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    began, told = time.monotonic(), False
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                held = os.pread(handle, 32, 0).decode("utf-8", "replace").strip() or "somebody"
                if not wait:
                    raise Busy(f"{path} is held by {held}") from exc
                if not told:
                    announce(f"waiting for {path.name}, held by {held}")
                    told = True
                if timeout and time.monotonic() - began > timeout:
                    raise Busy(f"{path} still held by {held} after {timeout:.0f}s") from exc
                time.sleep(0.5)
        os.ftruncate(handle, 0)
        os.pwrite(handle, f"pid {os.getpid()}".encode(), 0)
        yield path
    finally:
        try:
            os.ftruncate(handle, 0)
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)
