"""Runner entries as task evidence: the project scope of a store and the verified facts of an entry."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ml_stack.activity import reuse
from ml_stack.workspace import project

__all__ = ["evidence", "scope", "verified"]


def scope(root: Path) -> str:
    """A short stable name for the project that ``root`` belongs to; ``local`` outside any project."""
    found = project.describe(start=root)
    return hashlib.sha256(found["key"].encode()).hexdigest()[:16] if found.get("key") else "local"


def verified(entry_id: str, where: str, base: Path | None = None) -> dict[str, Any]:
    """The facts of runner entry ``entry_id`` in project scope ``where``; ValueError unless it verifies."""
    found = reuse.find(entry_id, where, base)
    if found is None:
        raise ValueError(f"no runner entry {entry_id[:24]!r} verifies in this project's store")
    return {"id": entry_id, "file": found["file"], "key": found["lookup"], "tree": found["tree"],
            "outcome": found["outcome"], "junit_sha256": found["junit_sha256"], "counts": found["counts"],
            "agent": found["agent"].get("id", ""), "created": found["created"]}


def evidence(entry_ids: object, base: Path | None = None) -> list[dict[str, Any]]:
    """The verified facts of the runner entries a worker cites, for the current project's store."""
    if not isinstance(entry_ids, list) or len(entry_ids) > 16 or not all(isinstance(i, str) for i in entry_ids):
        raise ValueError("test entries are a list of at most 16 runner entry ids")
    where = scope(Path.cwd())
    return [verified(entry_id, where, base) for entry_id in entry_ids]
