"""The machine-local cache the incremental checkers keep their results in."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from functools import cache
from pathlib import Path

FORCE = "POOLHOUSE_GATES_FULL"
"""Set to any value to ignore what is cached and compute every result afresh."""


def directory() -> Path:
    """Where the incremental results live, beside the whole-tree results."""
    return Path(tempfile.gettempdir()) / "poolhouse-gates" / "incremental"


def forced() -> bool:
    """True when the environment asks for a full run."""
    return bool(os.environ.get(FORCE))


def load(name: str) -> dict:
    """The stored object called ``name``, empty when it is missing, unreadable or not an object."""
    try:
        value = json.loads((directory() / f"{name}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def save(name: str, value: dict) -> None:
    """Store ``value`` as ``name`` by writing beside it and renaming, so a reader sees all or none."""
    place = directory()
    place.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=place, prefix=f"{name}-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(value, out)
        Path(temporary).replace(place / f"{name}.json")
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def digest(data: bytes) -> str:
    """The hash of some bytes."""
    return hashlib.sha256(data).hexdigest()


@cache
def _digest_at(path: Path, mtime_ns: int, size: int) -> str:
    try:
        return digest(path.read_bytes())
    except OSError:
        return ""


def file_digest(path: Path) -> str:
    """The hash of a file's bytes, empty when it cannot be read; one read per version of the file."""
    try:
        seen = path.stat()
    except OSError:
        return ""
    return _digest_at(path, seen.st_mtime_ns, seen.st_size)


def stamp(*files: Path) -> str:
    """A hash of the interpreter and of these source files, which changes when any code does."""
    parts = [sys.version, *(file_digest(path) for path in files)]
    return digest("\0".join(parts).encode())
