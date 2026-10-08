"""Read and verify the test runner's reuse entries: the entry file, its hash and its chain row."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ml_stack import home

__all__ = ["FIELDS", "SCHEMA", "chain_ok", "entry", "entry_hash", "find", "reuse_base", "row_hash", "rows"]

SCHEMA = 1
FIELDS = frozenset({"schema", "lookup", "file", "outcome", "manifest", "manifest_digest", "command",
                    "junit_sha256", "counts", "tree", "runner", "agent", "created", "prev", "entry_sha256"})
_verified: dict[str, tuple[int, int]] = {}


def _sha(value: dict[str, Any], skip: str) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in value.items() if k != skip}, sort_keys=True).encode()).hexdigest()


def entry_hash(value: dict[str, Any]) -> str:
    """The hash over every field of an entry except its own hash."""
    return _sha(value, "entry_sha256")


def row_hash(value: dict[str, Any]) -> str:
    """The hash over a chain row except its own hash."""
    return _sha(value, "row_sha256")


def reuse_base() -> Path:
    """The directory that holds one reuse store per project."""
    named = os.environ.get("DEV_TEST_REUSE_DIR")
    if named:
        return home.expand(named)
    slots = os.environ.get("DEV_TEST_SLOTS_DIR")
    return (home.expand(slots).resolve().parent if slots else home.user_home() / ".cache") / "test-reuse"


def rows(folder: Path) -> list[dict[str, Any]]:
    """The chain rows of a store, oldest first; empty when the chain is missing or unreadable."""
    try:
        return [json.loads(line) for line in (folder / "chain.jsonl").read_text(encoding="utf-8").splitlines() if line]
    except (OSError, ValueError):
        return []


def chain_ok(folder: Path) -> bool:
    """Whether every chain row hashes correctly and links to the one before it."""
    try:
        stat = (folder / "chain.jsonl").stat()
    except OSError:
        return True
    stamp = (stat.st_size, stat.st_mtime_ns)
    if _verified.get(str(folder)) != stamp:
        previous = ""
        for index, row in enumerate(rows(folder)):
            if row.get("seq") != index or row.get("prev") != previous or row.get("row_sha256") != row_hash(row):
                return False
            previous = row["row_sha256"]
        _verified[str(folder)] = stamp
    return True


def entry(folder: Path, entry_id: str) -> dict[str, Any] | None:
    """The entry with this id if its file, hash and chain row all agree, else None."""
    if not entry_id.isalnum() or not chain_ok(folder):
        return None
    try:
        found = json.loads((folder / "entries" / f"{entry_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(found, dict) or set(found) != FIELDS or found.get("schema") != SCHEMA \
            or found["entry_sha256"][:20] != entry_id or entry_hash(found) != found["entry_sha256"]:
        return None
    row = next((r for r in rows(folder) if r["id"] == entry_id), None)
    return found if row and row["entry_sha256"] == found["entry_sha256"] else None


def find(entry_id: str, scope: str, base: Path | None = None) -> dict[str, Any] | None:
    """The verified entry ``entry_id`` in project ``scope``'s store, or None."""
    return entry((base or reuse_base()) / scope, entry_id)
