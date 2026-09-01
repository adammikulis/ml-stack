"""JSON files that another process may be reading while they are rewritten.

A pipeline writes its state as it goes and something else — a page being served, a second
command, a person with ``cat`` — reads it at any moment. A plain ``open(..., "w")`` shows
that reader an empty file and then half of one. Everything here writes beside the file and
renames over it, so a reader sees the old contents or the new, never the middle.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Set
from pathlib import Path
from typing import Any

__all__ = ["prune_orphans", "read_json", "write_json"]


def write_json(path: Path, obj: Any, *, indent: int | None = 2) -> None:
    """Write ``obj`` as JSON so a concurrent reader sees the old file or the new one.

    The bytes go to a temporary file in the same directory and are renamed over ``path``;
    a failure part-way (a value that is not JSON, a full disk) leaves the old file as it was
    and no temporary behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=indent, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path, default: Any) -> Any:
    """What ``path`` holds, or ``default`` when it is missing or is not JSON."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def prune_orphans(directory: Path, live: Set[str], suffix: str = ".json") -> list[str]:
    """Delete every ``<id><suffix>`` in ``directory`` whose id is not in ``live``.

    For a directory of one file per record of a log — an extraction per message, a
    thumbnail per page — that would otherwise keep records the log has since dropped.
    Returns the ids that went, sorted. A directory that does not exist has no orphans.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    gone: list[str] = []
    for f in sorted(directory.glob(f"*{suffix}")):
        if f.stem not in live:
            f.unlink()
            gone.append(f.stem)
    return gone
