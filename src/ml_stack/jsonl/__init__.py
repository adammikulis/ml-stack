"""Reading, writing and compacting JSON-lines files."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Set
from pathlib import Path
from typing import Any

from ml_stack.files import write_text

__all__ = ["append", "compact", "read", "rows", "ts_key", "write"]


class MalformedLine(ValueError):
    """A line in a JSON-lines file is not JSON. Names the file and the line."""


def rows(path: Path | str) -> Iterator[Any]:
    """Each record in ``path``, in order. Blank lines are skipped; a bad line raises."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise MalformedLine(f"{path}:{number} is not JSON: {exc}") from None


def read(path: Path | str) -> list[Any]:
    """Every record in ``path``."""
    return list(rows(path))


def _dump(records: Iterable[Any], default: Callable[[Any], Any] | None) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, default=default) + "\n" for r in records)


def write(path: Path | str, records: Iterable[Any], *,
          default: Callable[[Any], Any] | None = None) -> int:
    """Write ``records`` as ``path``, replacing it. Returns how many were written."""
    body = _dump(records, default)
    write_text(path, body)
    return body.count("\n")


def append(path: Path | str, records: Iterable[Any], *,
           default: Callable[[Any], Any] | None = None) -> int:
    """Add ``records`` to the end of ``path``, creating it. Returns how many were added."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _dump(records, default)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(body)
    return body.count("\n")


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
    write_text(path, body)
    return (len(kept), len(lines) - len(kept))
