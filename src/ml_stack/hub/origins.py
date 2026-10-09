"""Where a model that came from a paired device came from, and whose licence acceptance it relied on.

An append-only local record (``<state>/onboard/peer-downloads.jsonl``, one JSON object per line,
never rewritten): when a file arrives from a peer, one line says who (the peer, by name and
certificate fingerprint), when, which file (name, size, sha256) and, for a gated model, the
acceptance of its licence on the serving device that this download relied on (who accepted and
when, as that device reported it). ``ml-stack-models list`` shows it. It is a record for the
person reading it, not an authority: nothing is decided from it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.platform import private_file

__all__ = ["latest", "read", "record"]


def _path(path: Path | str | None) -> Path:
    return Path(path) if path else home.state("onboard", "peer-downloads.jsonl")


def record(row: dict[str, Any], path: Path | str | None = None) -> dict[str, Any]:
    """Append ``row`` (with a ``at`` time) to the record and return what was written."""
    line = {"at": time.time(), **row}
    target = _path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
    private_file(target)
    return line


def read(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Every line of the record, oldest first; a line that is not an object is skipped."""
    try:
        text = _path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def latest(name: str, size: int, path: Path | str | None = None) -> dict[str, Any] | None:
    """The newest line for a file of this name and size, or None when it was not got from a peer."""
    for row in reversed(read(path)):
        if row.get("file") == name and row.get("size") == size:
            return row
    return None
