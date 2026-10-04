"""Boards, their members and each identity's subscriptions, replayed from ``boards.jsonl``."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack.files import read_json, write_json
from ml_stack.workspace import plain
from ml_stack.workspace.chain import ChainLog, held

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
    """The board events of one workspace: every state is a replay of the chained log."""

    def __init__(self, base: Path, clock: Callable[[], float] = time.time) -> None:
        self.base = base
        self.clock = clock
        self.log = ChainLog(base / "boards.jsonl", clock)
        self.guard = base / "boards.lock"

    def state(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[tuple[str, str], str]]]:
        """``(boards, subscriptions)``: boards by name with ``members``, subscriptions by
        identity mapping ``(type, target)`` to a delivery mode."""
        boards: dict[str, dict[str, Any]] = {GENERAL: {
            "by": "", "open": True, "project": "", "title": "everyone", "members": set(),
            "created": 0.0},
            ANNOUNCE: {"by": "", "open": True, "project": "", "title": "terse one-line progress",
                       "members": set(), "created": 0.0}}
        subs: dict[str, dict[tuple[str, str], str]] = {}
        for r in self.log.rows():
            op, name, who = r.get("op"), r.get("name", ""), r.get("who", "")
            if r.get("kind") == "board":
                if op == "create" and name not in boards:
                    boards[name] = {"by": r["by"], "open": bool(r["open"]),
                                    "project": r.get("project", ""), "title": r.get("title", ""),
                                    "members": {r["by"]} if r["by"] else set(),
                                    "created": r["ts"]}
                elif op == "join" and name in boards:
                    boards[name]["members"].add(who)
                elif op == "leave" and name in boards:
                    boards[name]["members"].discard(who)
            elif r.get("kind") == "sub":
                mine = subs.setdefault(who, {})
                key = (r["stype"], r["target"])
                if op == "set":
                    mine[key] = r["mode"]
                else:
                    mine.pop(key, None)
        return boards, subs

    def sinces(self) -> dict[str, dict[tuple[str, str], int]]:
        """Per identity, the message sequence number each subscription was made at: a
        subscription delivers only what arrives after it."""
        found: dict[str, dict[tuple[str, str], int]] = {}
        for r in self.log.rows():
            if r.get("kind") == "sub":
                mine = found.setdefault(r.get("who", ""), {})
                key = (r["stype"], r["target"])
                if r.get("op") == "set":
                    mine[key] = int(r.get("since", 0))
                else:
                    mine.pop(key, None)
        return found

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

    def _marks_file(self, who: str) -> Path:
        return self.base / "cursors" / f"{who.replace('/', '~')}.marks.json"

    def marks(self, who: str) -> dict[str, int]:
        """What ``who`` has read: the highest sequence number seen per board, DM or digest."""
        data = read_json(self._marks_file(who), {})
        found = data.get("marks", {}) if isinstance(data, dict) else {}
        return {str(k): int(v) for k, v in found.items()}

    def mark(self, who: str, key: str, seq: int) -> None:
        """Record that ``who`` has read ``key`` up to ``seq``; never moves backwards."""
        path = self._marks_file(who)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with held(path.with_name(path.name + ".lock")):
            now = self.marks(who)
            now[key] = max(now.get(key, 0), seq)
            write_json(path, {"version": VERSION, "marks": now})
