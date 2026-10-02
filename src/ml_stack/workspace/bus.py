"""The ordered message log: inboxes, outboxes, threads, read cursors and waiting."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack.files import read_json, write_json
from ml_stack.workspace.chain import ChainLog, held

__all__ = ["BROADCAST", "TYPES", "Bus", "mentions_me"]

BROADCAST = "*"
TYPES = ("task", "status", "handoff", "question", "answer", "claim", "release", "note")
VERSION = 1


def mentions_me(row: dict[str, Any], me: str) -> bool:
    """Whether ``row`` is addressed to ``me`` or to everyone, and was not sent by ``me``."""
    return row["from"] != me and row["to"] in (me, BROADCAST)


class Bus:
    """Every message in one chained log; an inbox is the rows addressed to an agent."""

    def __init__(self, base: Path, clock: Callable[[], float] = time.time) -> None:
        self.base = base
        self.clock = clock
        self.log = ChainLog(base / "bus.jsonl", clock)

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        """Add a message row; returns it with its sequence number."""
        return self.log.append({"kind": "msg", **row})

    def live(self, row: dict[str, Any]) -> bool:
        """Whether ``row`` has not passed its expiry."""
        return not row.get("expires") or float(row["expires"]) > self.clock()

    def get(self, seq: int) -> dict[str, Any] | None:
        """The message with sequence number ``seq``."""
        return next((r for r in self.log.rows() if r["seq"] == seq), None)

    def _cursor_file(self, me: str) -> Path:
        return self.base / "cursors" / f"{me}.json"

    def cursor(self, me: str) -> int:
        """The highest sequence number ``me`` acknowledged."""
        data = read_json(self._cursor_file(me), {})
        return int(data.get("seq", 0)) if isinstance(data, dict) else 0

    def ack(self, me: str, seq: int) -> int:
        """Record that ``me`` has handled everything up to ``seq``; never moves backwards."""
        path = self._cursor_file(me)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with held(path.with_name(path.name + ".lock")):
            now = max(self.cursor(me), seq)
            write_json(path, {"version": VERSION, "seq": now})
        return now

    def inbox(self, me: str, after: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Live messages for ``me`` after ``after`` (default: after the acknowledged cursor)."""
        start = self.cursor(me) if after is None else after
        found = [r for r in self.log.rows() if r["kind"] == "msg" and r["seq"] > start
                 and mentions_me(r, me) and self.live(r)]
        return found[:limit]

    def pending(self, me: str) -> int:
        """How many live messages ``me`` has not acknowledged."""
        return len(self.inbox(me, limit=1 << 30))

    def outbox(self, me: str, limit: int = 50) -> list[dict[str, Any]]:
        """The last ``limit`` messages ``me`` sent."""
        return [r for r in self.log.rows() if r["kind"] == "msg" and r["from"] == me][-limit:]

    def thread(self, root: int) -> list[dict[str, Any]]:
        """The message ``root`` and every reply under it, in order."""
        return [r for r in self.log.rows() if r["kind"] == "msg"
                and root in (r["seq"], r.get("thread"))]

    def wait(self, me: str, timeout: float, poll: float = 0.1) -> list[dict[str, Any]]:
        """Messages for ``me`` as soon as there are any, or none after ``timeout`` seconds."""
        deadline = time.monotonic() + timeout
        while True:
            found = self.inbox(me)
            if found or time.monotonic() >= deadline:
                return found
            time.sleep(min(poll, max(deadline - time.monotonic(), 0.0)))

    def prune(self, retention_s: float) -> int:
        """Drop the oldest messages that are past retention; returns how many."""
        horizon = self.clock() - retention_s
        return self.log.prune_prefix(lambda r: float(r["ts"]) < horizon)
