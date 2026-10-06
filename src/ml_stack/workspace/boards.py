"""Boards, memberships, subscriptions and read cursors in GraphStore."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack.workspace import plain
from ml_stack.workspace.chain import held
from ml_stack.workspace.graphlog import GraphLog

__all__ = ["ANNOUNCE", "ANNOUNCE_KINDS", "ANNOUNCE_MARK", "GENERAL", "MODES", "STYPES", "Boards",
           "project_board_name"]

GENERAL = "#general"
ANNOUNCE = "#announcements"
ANNOUNCE_KINDS = ("joined", "milestone", "done", "blocked")
ANNOUNCE_MARK = ANNOUNCE
MODES = ("inbox", "digest", "silent")
STYPES = ("board", "thread", "agent", "kind", "mentions")
VERSION = 1


def project_board_name(project: dict[str, str]) -> str:
    """The board name for a recorded project: ``#`` and its cleaned, lowercased name."""
    name = re.sub(r"[^a-z0-9._-]+", "-", project.get("name", "").lower()).strip("._-")
    return "#" + (name or "project")[:40]


class Boards:
    """Relational board state for one workspace."""

    def __init__(self, base: Path, clock: Callable[[], float] = time.time) -> None:
        self.base = base
        self.clock = clock
        self.log = GraphLog(base, "boards", clock)
        self.guard = base / "boards.lock"

    def state(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[tuple[str, str], str]]]:
        """``(boards, subscriptions)``: boards by name with ``members``, subscriptions by
        identity mapping ``(type, target)`` to a delivery mode."""
        boards, subs, _ = self.log.graph.state()
        return boards, subs

    def sinces(self) -> dict[str, dict[tuple[str, str], int]]:
        """Per identity, the message sequence number each subscription was made at: a
        subscription delivers only what arrives after it."""
        return self.log.graph.state()[2]

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        """Add one event row."""
        return self.log.append(row)

    def locked(self) -> Any:
        """Hold the boards lock for a check-then-write."""
        return held(self.guard)

    def project_board(self, project: dict[str, str]) -> str:
        """The board for ``project`` (creating it when there is none), by its key."""
        key = project.get("key", "")
        with self.locked():
            boards, _ = self.state()
            for name, b in boards.items():
                if key and b["project"] == key:
                    return name
            name = project_board_name(project)
            if name in boards:
                name = f"{name[:33]}-{hashlib.sha256(key.encode()).hexdigest()[:6]}"
            if name in boards:
                return name
            self.append({"kind": "board", "op": "create", "name": name, "by": "", "open": False,
                         "project": key, "title": plain.line(project.get("name", ""), 60)})
            return name

    def marks(self, who: str) -> dict[str, int]:
        """What ``who`` has read: the highest sequence number seen per board, DM or digest."""
        return self.log.graph.marks(who)

    def mark(self, who: str, key: str, seq: int) -> None:
        """Record that ``who`` has read ``key`` up to ``seq``; never moves backwards."""
        self.log.graph.cursor(who, key, seq)
