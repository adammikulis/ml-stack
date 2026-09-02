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

The lock is `flock` on POSIX and `msvcrt.locking` (`LockFile`) on Windows, chosen when the
lock is taken rather than when this module is imported, so the module loads on either and a
test on one can stand in for the other. Both release the lock when the holder's handle
closes -- including when the holder dies -- which is what makes a stale lock file harmless.
The one difference worth knowing: a `LockFile` region is *mandatory*, so a byte another
process has locked cannot even be read. The pid is written at offset 0 for a person looking
at a stalled machine to see, so the byte that is locked is one far past it.
"""

from __future__ import annotations

import errno
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

__all__ = ["Busy", "only_one"]

#: The byte locked on Windows. Past any pid text, so a reader is never refused by the lock.
_LOCKED_BYTE = 1 << 30


class Busy(RuntimeError):
    """Somebody else holds the lock and we were told not to wait."""


# -- the two ways of holding a byte -----------------------------------------------------
def _try_take(handle: int) -> bool:
    """Take the lock without waiting. True when held now, False when somebody else has it."""
    if sys.platform == "win32":
        import msvcrt

        os.lseek(handle, _LOCKED_BYTE, os.SEEK_SET)
        try:
            msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            # EACCES is what LockFile's refusal comes back as; EDEADLOCK is what the
            # blocking mode would say, and is kept in case a fake or a future runtime does.
            if exc.errno in (errno.EACCES, errno.EAGAIN, getattr(errno, "EDEADLOCK", 36)):
                return False
            raise
    import fcntl

    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EACCES):
            return False
        raise


def _drop(handle: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(handle, _LOCKED_BYTE, os.SEEK_SET)
        msvcrt.locking(handle, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle, fcntl.LOCK_UN)


def _holder(handle: int) -> str:
    """Who has it, as written at offset 0. ``os.pread`` does not exist on Windows."""
    os.lseek(handle, 0, os.SEEK_SET)
    try:
        return os.read(handle, 32).decode("utf-8", "replace").strip() or "somebody"
    except OSError:
        return "somebody"


def _write_holder(handle: int) -> None:
    os.ftruncate(handle, 0)
    os.lseek(handle, 0, os.SEEK_SET)
    os.write(handle, f"pid {os.getpid()}".encode())


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
        while not _try_take(handle):
            held = _holder(handle)
            if not wait:
                raise Busy(f"{path} is held by {held}")
            if not told:
                announce(f"waiting for {path.name}, held by {held}")
                told = True
            if timeout and time.monotonic() - began > timeout:
                raise Busy(f"{path} still held by {held} after {timeout:.0f}s")
            time.sleep(0.5)
        _write_holder(handle)
        yield path
    finally:
        try:
            os.ftruncate(handle, 0)
            _drop(handle)
        finally:
            os.close(handle)
