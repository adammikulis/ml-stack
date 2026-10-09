"""A built test artifact (a wheel, a seeded graph store) made once per source state.

Several fixtures spend 10 to 40 s building something every worker and every run builds again
with the same result: the inputs are the sources that make it. ``cached`` keeps the result
under the temporary directory by a fingerprint of those inputs (path, size, modification time),
so a worker, or a later run, finds the finished directory and builds nothing; a changed input
builds a new one. The directory is shared and must be treated as read-only.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path

KEEP_SECONDS = 86400


def expand(*paths: Path) -> list[Path]:
    """Every file under the given files and directories, caches excluded, in a stable order."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found += sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        else:
            found.append(path)
    return found


def fingerprint(files: Iterable[Path]) -> str:
    """A digest of each file's path, size and modification time."""
    digest = hashlib.sha256()
    for path in files:
        try:
            info = path.stat()
        except OSError:
            continue
        digest.update(f"{path}:{info.st_size}:{info.st_mtime_ns}\n".encode())
    return digest.hexdigest()[:20]


def _forget_old(base: Path) -> None:
    for entry in base.iterdir():
        try:
            if time.time() - entry.stat().st_mtime > KEEP_SECONDS:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass


def cached(name: str, inputs: Iterable[Path], build: Callable[[Path], None]) -> Path:
    """The directory ``build(directory)`` fills, built on first use for these ``inputs`` and found
    afterwards. Two workers that build at once each fill their own staging directory; the first
    to finish publishes it and the other discards its copy."""
    owner = os.getuid() if hasattr(os, "getuid") else 0
    base = Path(tempfile.gettempdir()) / f"ml-stack-test-cache-{owner}" / name
    base.mkdir(parents=True, exist_ok=True)
    final = base / fingerprint(inputs)
    if not final.is_dir():
        _forget_old(base)
        staging = Path(tempfile.mkdtemp(prefix="building-", dir=base))
        build(staging)
        try:
            staging.rename(final)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
    return final
