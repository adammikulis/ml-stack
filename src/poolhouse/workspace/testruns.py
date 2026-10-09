"""Runner entries as task evidence: the project scope of a store and the verified facts of an entry."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from poolhouse import home
from poolhouse.activity import reuse
from poolhouse.memory import project as repository
from poolhouse.net import git
from poolhouse.workspace import project

__all__ = ["STORE_BASE", "evidence", "scope", "task_checkout", "task_scope", "verified"]

STORE_BASE = home.user_home() / ".cache" / "test-reuse"


def scope(root: Path) -> str:
    """A short stable name for the project that ``root`` belongs to, shared by every worktree of one
    repository (by origin url, else by the common git directory); a checkout that is no repository
    is its own scope, keyed by its resolved path."""
    found = project.describe(start=root)
    shared = repository.common_git_dir(root) if repository.at(root).ident.startswith("path:") else None
    if shared is not None:
        return "repo-" + hashlib.sha256(str(shared).encode()).hexdigest()[:12]
    if found.get("key"):
        return hashlib.sha256(found["key"].encode()).hexdigest()[:16]
    return "local-" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]


def task_checkout(graph: Any, ident: str) -> Path:
    """The checkout the task board assigned to ``ident``."""
    attrs = next((n["attrs"] for n in graph.nodes("task-worktree") if n["attrs"].get("task") == ident), None)
    if attrs is None:
        raise ValueError("the task has no assigned checkout, so its project is unknown")
    return Path(attrs.get("project") or attrs["source_project"])


def task_scope(graph: Any, ident: str) -> str:
    """The project scope of the checkout the task board assigned to ``ident``."""
    return scope(task_checkout(graph, ident))


def verified(entry_id: str, where: str, base: Path | None = None, commit: str = "",
             repo: Path | None = None) -> dict[str, Any]:
    """The facts of runner entry ``entry_id`` in project scope ``where``; ValueError unless it verifies,
    passed, ran on ``commit`` in a clean checkout, and holds the tree ``commit`` has in ``repo``."""
    found = reuse.find(entry_id, where, base or STORE_BASE)
    label = f"runner entry {entry_id[:24]!r}"
    if found is None:
        raise ValueError(f"no {label} verifies in this project's store")
    if found["outcome"] != "pass":
        raise ValueError(f"{label} did not pass")
    if not commit:
        raise ValueError(f"{label} is evidence for a named commit; give the commit it ran on")
    if found["commit"] != commit:
        raise ValueError(f"{label} ran on commit {found['commit'][:12] or 'unknown'}, not {commit[:12]}")
    if not found["clean"]:
        raise ValueError(f"{label} ran with uncommitted changes, so it is not evidence for a commit")
    try:
        tree = git.run(["rev-parse", f"{commit}^{{tree}}"], cwd=repo).stdout.strip() if repo else ""
    except git.GitFailed as error:
        raise ValueError(f"{label}: the commit's tree cannot be read ({error})") from error
    if tree != found["tree"]:
        raise ValueError(f"{label} ran on tree {found['tree'][:12]}, not the tree of commit {commit[:12]}")
    return {"id": entry_id, "file": found["file"], "key": found["lookup"], "tree": found["tree"],
            "commit": found["commit"], "outcome": found["outcome"], "junit_sha256": found["junit_sha256"],
            "counts": found["counts"], "agent": found["agent"].get("id", ""), "created": found["created"],
            "covers": f"{found['file']} only"}


def evidence(entry_ids: object, graph: Any, ident: str, commit: str = "") -> list[dict[str, Any]]:
    """The verified facts of the runner entries a worker cites for task ``ident``, from its project's store."""
    if not isinstance(entry_ids, list) or len(entry_ids) > 16 or not all(isinstance(i, str) for i in entry_ids):
        raise ValueError("test entries are a list of at most 16 runner entry ids")
    if not entry_ids:
        return []
    checkout = task_checkout(graph, ident)
    return [verified(entry_id, scope(checkout), None, commit, checkout) for entry_id in entry_ids]
