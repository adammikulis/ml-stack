"""Per-agent wake-ups over named pipes, with jittered backoff where pipes are not available."""

from __future__ import annotations

import contextlib
import errno
import os
import random
import select
import stat
import sys
import time
from pathlib import Path

__all__ = ["BACKOFF_MAX_S", "BACKOFF_MIN_S", "Waiter", "backoff", "signal"]

BACKOFF_MIN_S = 0.1
BACKOFF_MAX_S = 2.0
_JITTER = random.SystemRandom()
PIPES = hasattr(os, "mkfifo") and sys.platform != "win32"


def backoff(step: int) -> float:
    """The sleep before re-checking after ``step`` empty checks: doubles from 0.1 s to 2 s, jittered."""
    return min(BACKOFF_MAX_S, BACKOFF_MIN_S * 2 ** max(step, 0)) * _JITTER.uniform(0.75, 1.25)


def _pipe(directory: Path, agent: str) -> Path:
    return directory / f"{agent}.fifo"


def _safe(agent: str) -> bool:
    return bool(agent) and agent[0] != "." and not any(c in agent for c in "/\\\0")


def signal(directory: Path, agents: list[str] | None = None) -> int:
    """Wake the waiters of ``agents`` (everyone when None); returns how many pipes had a reader."""
    if not PIPES or not directory.is_dir():
        return 0
    names = [f"{a}.fifo" for a in agents if _safe(a)] if agents is not None else sorted(
        p.name for p in directory.iterdir() if p.name.endswith(".fifo"))
    woken = 0
    for name in names:
        try:
            fd = os.open(directory / name, os.O_WRONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        except OSError:
            continue
        try:
            if stat.S_ISFIFO(os.fstat(fd).st_mode):
                with contextlib.suppress(OSError):
                    os.write(fd, b"1")
                    woken += 1
        finally:
            os.close(fd)
    return woken


class Waiter:
    """One agent's wake-up: open before checking the inbox, then `sleep` between checks."""

    def __init__(self, directory: Path, agent: str) -> None:
        self.fd = -1
        self.step = 0
        if not PIPES:
            return
        path = _pipe(directory, agent)
        if not _safe(agent):
            return
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not path.exists():
                with contextlib.suppress(FileExistsError):
                    os.mkfifo(path, 0o600)
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        except OSError as exc:
            if exc.errno not in (errno.ENOENT, errno.ELOOP, errno.ENXIO, errno.EACCES, errno.ENOTDIR):
                raise
            return
        if stat.S_ISFIFO(os.fstat(fd).st_mode):
            self.fd = fd
        else:
            os.close(fd)

    def sleep(self, seconds: float) -> None:
        """Return when signalled or after about ``seconds`` (at most the backoff for this step)."""
        self.step += 1
        wait = min(seconds, backoff(self.step))
        if self.fd < 0:
            time.sleep(wait)
            return
        ready, _, _ = select.select([self.fd], [], [], wait)
        if ready:
            self.step = 0
            with contextlib.suppress(OSError):
                os.read(self.fd, 4096)

    def close(self) -> None:
        """Release the pipe."""
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
