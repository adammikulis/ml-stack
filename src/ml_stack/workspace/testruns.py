"""Runner entries as task evidence: the project scope of a store and the verified facts of an entry."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.activity import reuse
from ml_stack.workspace import project

__all__ = ["STORE_BASE", "evidence", "scope", "task_scope", "verified"]

STORE_BASE = home.user_home() / ".cache" / "test-reuse"


def scope(root: Path) -> str:
    """A short stable name for the project that ``root`` belongs to; a checkout that names no project
    is its own scope, keyed by its resolved path."""
    found = project.describe(start=root)
    if found.get("key"):
        return hashlib.sha256(found["key"].encode()).hexdigest()[:16]
    return "local-" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]


def task_scope(graph: Any, ident: str) -> str:
    """The project scope of the checkout the task board assigned to ``ident``."""
    attrs = next((n["attrs"] for n in graph.nodes("task-worktree") if n["attrs"].get("task") == ident), None)
    if attrs is None:
        raise ValueError("the task has no assigned checkout, so its project is unknown")
    return scope(Path(attrs.get("project") or attrs["source_project"]))


def verified(entry_id: str, where: str, base: Path | None = None, commit: str = "") -> dict[str, Any]:
    """The facts of runner entry ``entry_id`` in project scope ``where``; ValueError unless it verifies,
    passed, and ran on ``commit`` when one is given."""
    found = reuse.find(entry_id, where, base or STORE_BASE)
    if found is None:
        raise ValueError(f"no runner entry {entry_id[:24]!r} verifies in this project's store")
    if found["outcome"] != "pass":
        raise ValueError(f"runner entry {entry_id[:24]!r} did not pass")
    if commit and found["commit"] != commit:
        raise ValueError(f"runner entry {entry_id[:24]!r} ran on commit {found['commit'][:12] or 'unknown'}, "
                         f"not {commit[:12]}")
    return {"id": entry_id, "file": found["file"], "key": found["lookup"], "tree": found["tree"],
            "commit": found["commit"], "outcome": found["outcome"], "junit_sha256": found["junit_sha256"],
            "counts": found["counts"], "agent": found["agent"].get("id", ""), "created": found["created"]}


def evidence(entry_ids: object, graph: Any, ident: str, commit: str = "") -> list[dict[str, Any]]:
    """The verified facts of the runner entries a worker cites for task ``ident``, from its project's store."""
    if not isinstance(entry_ids, list) or len(entry_ids) > 16 or not all(isinstance(i, str) for i in entry_ids):
        raise ValueError("test entries are a list of at most 16 runner entry ids")
    if not entry_ids:
        return []
    where = task_scope(graph, ident)
    return [verified(entry_id, where, None, commit) for entry_id in entry_ids]
