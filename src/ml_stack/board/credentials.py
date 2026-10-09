"""What a client keeps for its sessions: a token file per board and name, and read cursors.

The node stores only the hash of a token; the token itself is in a file only its owner reads.
A cursor is the map the node returned for one question (the inbox, the announcements), so the
next read of that question returns only what is new.
"""

from __future__ import annotations

from pathlib import Path

from ml_stack.board.client import Denied
from ml_stack.files import read_json, write_json, writing
from ml_stack.private_path import problem

__all__ = ["cursors", "forget", "load", "remember", "store"]


def _folder(state: Path, board: str) -> Path:
    path = state / "client" / board
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    for parent in (path.parent, path):
        parent.chmod(0o700)
    return path


def store(state: Path, board: str, name: str, token: str) -> Path:
    """Write ``token`` for ``name`` on ``board``, atomically and readable by this user only."""
    target = _folder(state, board) / f"{name}.token"
    with writing(target) as tmp:
        tmp.write_text(token + "\n", encoding="utf-8")
        tmp.chmod(0o600)
    return target


def load(state: Path, board: str, name: str) -> str:
    """``name``'s token on ``board``; `Denied` when there is none or its file is not private."""
    path = state / "client" / board / f"{name}.token"
    why = problem(path)
    if why:
        raise Denied("denied", f"no usable token for {name} on {board}: {path} {why}")
    return path.read_text(encoding="utf-8").strip()


def cursors(state: Path, board: str, name: str) -> dict[str, dict[str, int]]:
    """The saved cursors of ``name``, by question."""
    found = read_json(state / "client" / board / f"{name}.cursors.json", {})
    return found if isinstance(found, dict) else {}


def remember(state: Path, board: str, name: str, question: str, cursor: dict[str, int]) -> None:
    """Keep ``cursor`` as the place ``name`` has read up to for ``question``."""
    saved = cursors(state, board, name)
    saved[question] = cursor
    target = _folder(state, board) / f"{name}.cursors.json"
    write_json(target, saved)
    target.chmod(0o600)


def forget(state: Path, board: str, name: str) -> None:
    """Delete the token file of a session whose token the node revoked."""
    (state / "client" / board / f"{name}.token").unlink(missing_ok=True)
