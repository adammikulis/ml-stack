"""Bounding the directory server logs are written to: a count, a total size and an age."""

from __future__ import annotations

import os
import time
from collections.abc import Collection
from pathlib import Path

FILES_ENV = "ML_STACK_LOG_FILES"
MEGABYTES_ENV = "ML_STACK_LOG_MB"
DAYS_ENV = "ML_STACK_LOG_DAYS"
FILES, MEGABYTES, DAYS = 60, 256, 30


def _number(name: str, fallback: float) -> float:
    try:
        return max(0.0, float(os.environ[name]))
    except (KeyError, ValueError):
        return fallback


def limits() -> tuple[int, int, float]:
    """``(files, bytes, seconds)`` the logs may take; 0 for any of them means no limit."""
    return (int(_number(FILES_ENV, FILES)), int(_number(MEGABYTES_ENV, MEGABYTES) * (1 << 20)),
            _number(DAYS_ENV, DAYS) * 86400)


def prune(directory: Path, keep: Collection[Path] = ()) -> list[Path]:
    """Delete the oldest ``*.log`` files in ``directory`` until the count, the total size and
    the age are within `limits`; a file in ``keep`` is never deleted. Returns what went."""
    most_files, most_bytes, most_age = limits()
    protected = {Path(one).resolve() for one in keep}
    found = sorted((one for one in directory.glob("*.log") if one.is_file()),
                   key=lambda one: (one.stat().st_mtime, one.name))
    gone: list[Path] = []
    now = time.time()
    total = sum(one.stat().st_size for one in found)
    for old in list(found):
        over = (most_files and len(found) - len(gone) > most_files) \
            or (most_bytes and total > most_bytes) \
            or (most_age and now - old.stat().st_mtime > most_age)
        if not over:
            break
        if old.resolve() in protected:
            continue
        size = old.stat().st_size
        old.unlink(missing_ok=True)
        total -= size
        gone.append(old)
    return gone
