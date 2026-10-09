"""What the daemon does for the unit that runs it: hold the single-instance lock and keep its log files small."""

from __future__ import annotations

import shutil
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path

from poolhouse import home
from poolhouse.lock import Busy, only_one

from .autostart_prepare import LOG_BYTES, LOG_KEEP

__all__ = ["rotate", "rotate_all", "running"]

CHECK_S = 600


def rotate(path: Path, max_bytes: int = LOG_BYTES, keep: int = LOG_KEEP) -> bool:
    """Copy ``path`` to ``path.1`` and empty it once it passes ``max_bytes``, keeping ``keep`` older copies."""
    try:
        if path.stat().st_size <= max_bytes:
            return False
        for number in range(keep - 1, 0, -1):
            older = path.with_name(f"{path.name}.{number}")
            if older.exists():
                shutil.copyfile(older, path.with_name(f"{path.name}.{number + 1}"))
        shutil.copyfile(path, path.with_name(f"{path.name}.1"))
        with path.open("r+b") as handle:
            handle.truncate(0)
    except OSError:
        return False
    return True


def rotate_all() -> int:
    """Rotate every log under the autostart log directory; how many were rotated."""
    logs = home.state("autostart", "logs")
    return sum(rotate(p) for p in sorted(logs.glob("*.log"))) if logs.is_dir() else 0


def _watch(stop: threading.Event) -> None:
    while True:
        with suppress(OSError):
            rotate_all()
        if stop.wait(CHECK_S):
            return


@contextmanager
def running(root: Path) -> Iterator[bool]:
    """Hold ``root``'s daemon lock for the block, rotating logs meanwhile; yields False when another daemon holds it."""
    with ExitStack() as stack:
        try:
            stack.enter_context(only_one(root / "traind.lock", wait=False, note="poolhouse-traind"))
        except Busy:
            yield False
            return
        stop = threading.Event()
        threading.Thread(target=_watch, args=(stop,), name="autostart-log-rotation", daemon=True).start()
        stack.callback(stop.set)
        yield True
