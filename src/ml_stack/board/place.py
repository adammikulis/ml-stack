"""Which board a client is on: the project the directory it runs in belongs to."""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import subprocess
from pathlib import Path

from ml_stack.board.client import Client, Denied, Invalid

__all__ = ["BOARD_ENV", "board_id", "resolve"]

BOARD_ENV = "ML_STACK_BOARD"


def _git(path: Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def board_id(path: Path) -> tuple[str, Path] | None:
    """``(board id, main working tree)`` of the repository ``path`` is in, or None. Every worktree
    of one repository gives the same id: its name and a few hex of its git common directory."""
    common = _git(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if not common:
        return None
    common_dir = Path(common).resolve()
    tree = common_dir.parent if common_dir.name == ".git" else common_dir
    word = re.sub(r"[^a-z0-9]+", "-", tree.name.lower()).strip("-")[:30] or "project"
    return f"{word}-{hashlib.sha256(str(common_dir).encode()).hexdigest()[:6]}", tree


def resolve(client: Client, path: Path | None = None, named: str = "") -> str:
    """The board for ``named``, else $ML_STACK_BOARD, else the project of ``path`` (default the
    working directory); a repository the node does not know yet is added as a project."""
    explicit = named or os.environ.get(BOARD_ENV, "")
    if explicit:
        return explicit
    where = (path or Path.cwd()).resolve()
    try:
        return str(client.call("project_resolve", path=str(where))["board"])
    except Denied:
        found = board_id(where)
        if found is None:
            raise
    board, tree = found
    with contextlib.suppress(Invalid):
        client.call("project_add", id=board, kind="git", path=str(tree))
    return str(client.call("project_resolve", path=str(where))["board"])
