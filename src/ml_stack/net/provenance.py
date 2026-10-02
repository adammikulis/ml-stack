"""Where a file came from, kept beside it and in an index of recent downloads."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml_stack import files, home

__all__ = ["HEADERS", "VERSION", "Provenance", "index_path", "recent", "record", "sidecar"]

VERSION = 1
HEADERS = ("content-type", "content-length", "content-disposition", "etag", "last-modified",
           "server", "date", "accept-ranges", "x-repo-commit", "x-linked-etag", "x-linked-size")
"""The response headers worth keeping; cookies and credentials are never among them."""
INDEX_LIMIT = 2000
_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class Provenance:
    """The record of one download: its source, how it got here, what was checked."""

    url: str
    final_url: str
    path: str
    sha256: str
    size: int
    kind: str
    fetched_at: str
    status: int = 200
    redirects: tuple[str, ...] = ()
    headers: Mapping[str, str] = field(default_factory=dict)
    expected_sha256: str = ""
    host_source: str = ""
    scan: str = ""
    scanners: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    outcome: str = "kept"
    reason: str = ""

    def as_json(self) -> dict[str, Any]:
        """The record as plain values, versioned."""
        return files.versioned({k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in asdict(self).items()}, VERSION)


def stamp() -> str:
    """The current time as an ISO-8601 UTC string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def subset(headers: Mapping[str, str]) -> dict[str, str]:
    """The headers in `HEADERS`, lower-cased, each cut to 200 characters."""
    low = {k.lower(): v for k, v in headers.items()}
    return {k: low[k][:200] for k in HEADERS if k in low}


def sidecar(path: Path) -> Path:
    """Where the provenance of ``path`` is kept."""
    return path.with_name(path.name + ".provenance.json")


def index_path() -> Path:
    """The append-only index of recent downloads, kept or not."""
    return home.state("net", "downloads.jsonl")


def record(entry: Provenance, *, beside: Path | None = None, index: Path | None = None) -> None:
    """Write ``entry`` beside the artifact (when it was kept) and append it to the index."""
    if beside is not None:
        files.write_json(sidecar(beside), entry.as_json())
    where = index or index_path()
    with _LOCK:
        where.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with where.open("a", encoding="utf-8") as out:
            out.write(json.dumps(entry.as_json(), sort_keys=True) + "\n")
        _trim(where)


def _trim(where: Path) -> None:
    try:
        lines = where.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) > INDEX_LIMIT:
        files.write_text(where, "\n".join(lines[-INDEX_LIMIT:]) + "\n")


def recent(limit: int = 20, index: Path | None = None) -> list[dict[str, Any]]:
    """The last ``limit`` index rows, newest first."""
    try:
        lines = (index or index_path()).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
        if len(rows) >= limit:
            break
    return rows
