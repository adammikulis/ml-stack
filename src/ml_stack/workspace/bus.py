"""The ordered message log: inboxes, outboxes, threads, read cursors and waiting."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack.files import read_json, write_json
from ml_stack.workspace import wake
from ml_stack.workspace.chain import ChainLog, held

__all__ = ["BROADCAST", "TYPES", "Bus", "mentions_me"]

BROADCAST = "*"
TYPES = ("task", "status", "handoff", "question", "answer", "claim", "release", "note")
VERSION = 1
CANCEL_SLICE_S = 0.25


def mentions_me(row: dict[str, Any], me: str) -> bool:
    """Whether ``row`` is addressed to ``me`` or to everyone, and was not sent by ``me``."""
    return row["from"] != me and row["to"] in (me, BROADCAST)


class _Index:
    """Thread and sender lookups over the rows seen so far; rebuilt from the log, never kept apart from it."""

    __slots__ = ("hash", "sent", "seq", "threads")

    def __init__(self) -> None:
        self.seq, self.hash = 0, ""
        self.threads: dict[int, list[int]] = {}
        self.sent: dict[str, list[int]] = {}

    def add(self, row: dict[str, Any]) -> None:
        """Index one row."""
        self.seq, self.hash = row["seq"], row["hash"]
        if row["kind"] != "msg":
            return
        self.sent.setdefault(row["from"], []).append(row["seq"])
        for root in {row["seq"], row.get("thread")} - {None, 0}:
            self.threads.setdefault(root, []).append(row["seq"])


class Bus:
    """Every message in one chained log; an inbox is the rows addressed to an agent."""

    def __init__(self, base: Path, clock: Callable[[], float] = time.time) -> None:
        self.base = base
        self.clock = clock
        self.log = ChainLog(base / "bus.jsonl", clock)
        self._index = _Index()

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        """Add a message row, wake its recipient's waiters; returns it with its sequence number."""
        made = self.log.append({"kind": "msg", **row})
        wake.signal(self.base / "wake", None if made["to"] == BROADCAST else [made["to"]])
        return made

    def live(self, row: dict[str, Any]) -> bool:
        """Whether ``row`` has not passed its expiry."""
        return not row.get("expires") or float(row["expires"]) > self.clock()

    def get(self, seq: int) -> dict[str, Any] | None:
        """The message with sequence number ``seq``."""
        found = self.log.after(seq - 1)[:1]
        return found[0] if found and found[0]["seq"] == seq else None

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
        found: list[dict[str, Any]] = []
        for r in self.log.after(start):
            if r["kind"] == "msg" and mentions_me(r, me) and self.live(r):
                found.append(r)
                if len(found) >= limit:
                    break
        return found

    def pending(self, me: str) -> int:
        """How many live messages ``me`` has not acknowledged."""
        return len(self.inbox(me, limit=1 << 30))

    def _indexed(self) -> _Index:
        """The thread and sender indexes, brought up to the verified log."""
        index = self._index
        if index.seq:
            anchor = self.get(index.seq)
            if anchor is None or anchor["hash"] != index.hash:
                index = self._index = _Index()
        for r in self.log.after(index.seq):
            index.add(r)
        return index

    def outbox(self, me: str, limit: int = 50) -> list[dict[str, Any]]:
        """The last ``limit`` messages ``me`` sent."""
        return self._rows(self._indexed().sent.get(me, [])[-limit:])

    def thread(self, root: int) -> list[dict[str, Any]]:
        """The message ``root`` and every reply under it, in order."""
        return self._rows(self._indexed().threads.get(root, []))

    def _rows(self, seqs: list[int]) -> list[dict[str, Any]]:
        if not seqs:
            return []
        rows = self.log.after(seqs[0] - 1)
        first = rows[0]["seq"] if rows else 0
        return [rows[s - first] for s in seqs if 0 <= s - first < len(rows)]

    def wait(self, me: str, timeout: float,
             cancel: Callable[[], bool] | None = None) -> list[dict[str, Any]]:
        """Messages for ``me`` as soon as there are any; none after ``timeout`` seconds or once
        ``cancel()`` is true."""
        deadline = time.monotonic() + timeout
        waiter = wake.Waiter(self.base / "wake", me)
        try:
            while True:
                found = self.inbox(me)
                left = deadline - time.monotonic()
                if found or left <= 0 or (cancel is not None and cancel()):
                    return found
                waiter.sleep(left if cancel is None else min(left, CANCEL_SLICE_S))
        finally:
            waiter.close()

    def prune(self, retention_s: float) -> int:
        """Drop the oldest messages that are past retention; returns how many."""
        horizon = self.clock() - retention_s
        return self.log.prune_prefix(lambda r: float(r["ts"]) < horizon)
