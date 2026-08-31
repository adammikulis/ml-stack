"""Compaction for JSON-lines files."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Set
from pathlib import Path
from typing import Any

__all__ = ["compact", "ts_key"]


def ts_key(ts: str) -> tuple[int, int] | None:
    """A "seconds.fraction" timestamp as (seconds, fraction); None if not numeric."""
    sec, _, frac = ts.partition(".")
    try:
        return (int(sec), int(frac or "0"))
    except ValueError:
        return None


def compact(path: Path, key: Callable[[Any], str | None], *,
            drop: Set[str] = frozenset(),
            order: Callable[[Any], Any] | None = None) -> tuple[int, int]:
    """Keep one line per key — the greatest by ``order``, the later line on a tie — and
    drop every line whose key is in ``drop`` or is None. Returns (kept, dropped)."""
    if not path.exists():
        return (0, 0)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    best: dict[str, tuple[Any, ...]] = {}
    kept: dict[str, str] = {}
    first_seen: list[str] = []
    for position, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        k = key(row)
        if k is None or k in drop:
            continue
        rank = (order(row), position) if order else (position,)
        if k not in kept:
            first_seen.append(k)
        if k not in kept or rank >= best[k]:
            kept[k] = line
            best[k] = rank
    if len(kept) == len(lines):
        return (len(kept), 0)
    body = "\n".join(kept[k] for k in first_seen) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
    return (len(kept), len(lines) - len(kept))
