"""An append-only JSON-lines log whose rows are numbered and hash-chained."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack import lock
from ml_stack.files import write_text

__all__ = ["GENESIS", "ChainBroken", "ChainLog", "Verdict", "held"]

GENESIS = "0" * 64
VERSION = 1


class ChainBroken(RuntimeError):
    """A row does not follow from the one before it."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a log's chain holds, how many rows it has, and where it first breaks."""

    ok: bool
    rows: int
    head: str
    broken_at: int = 0
    reason: str = ""


@contextmanager
def held(path: Path) -> Iterator[None]:
    """Hold ``path`` exclusively for the block, polling every few milliseconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        while not lock.take(fd):
            time.sleep(0.005)
        try:
            yield
        finally:
            lock.release(fd)
    finally:
        os.close(fd)


def _digest(prev: str, row: dict[str, Any]) -> str:
    body = json.dumps({k: v for k, v in row.items() if k != "hash"}, sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256((prev + body).encode()).hexdigest()


class ChainLog:
    """Rows with ``seq``, ``prev`` and ``hash``, fsynced on append.

    A final line without a newline (a crash mid-write) is cut off the next time the log is
    appended to, and the cut bytes are kept in ``<name>.torn``.
    """

    def __init__(self, path: Path | str, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.clock = clock
        self.guard = self.path.with_name(self.path.name + ".lock")

    def _lines(self) -> tuple[list[bytes], bytes]:
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            return [], b""
        whole, _, torn = data.rpartition(b"\n")
        return ([ln for ln in whole.split(b"\n") if ln.strip()] if whole or data.endswith(b"\n")
                else []), torn

    def _walk(self) -> tuple[list[dict[str, Any]], Verdict]:
        lines, _ = self._lines()
        rows: list[dict[str, Any]] = []
        prev, base = GENESIS, 0
        for number, line in enumerate(lines, start=1):
            try:
                row = json.loads(line)
            except ValueError:
                return rows, Verdict(False, len(rows), prev, number, "row is not JSON")
            if not isinstance(row, dict):
                return rows, Verdict(False, len(rows), prev, number, "row is not an object")
            if number == 1 and "base" in row:
                prev, base = str(row["base"]["hash"]), int(row["base"]["seq"])
                continue
            if row.get("prev") != prev or row.get("hash") != _digest(prev, row):
                return rows, Verdict(False, len(rows), prev, number, "hash does not follow")
            if row.get("seq") != base + len(rows) + 1:
                return rows, Verdict(False, len(rows), prev, number, "sequence number skips")
            rows.append(row)
            prev = row["hash"]
        return rows, Verdict(True, len(rows), prev)

    def rows(self) -> list[dict[str, Any]]:
        """The rows whose chain holds, in order."""
        return self._walk()[0]

    def verify(self, anchor: str = "") -> Verdict:
        """Whether the chain holds, and when ``anchor`` is a head hash, that it is still in it."""
        rows, verdict = self._walk()
        if verdict.ok and anchor and anchor not in {GENESIS, *(r["hash"] for r in rows)}:
            return Verdict(False, verdict.rows, verdict.head, 0, "anchor is not in the log")
        return verdict

    def head(self) -> str:
        """The hash of the last row."""
        return self._walk()[1].head

    def append(self, body: dict[str, Any]) -> dict[str, Any]:
        """Add a row made of ``body``; returns it with its ``seq``, ``prev`` and ``hash``."""
        with held(self.guard):
            self._cut_torn()
            rows, verdict = self._walk()
            if not verdict.ok:
                raise ChainBroken(f"{self.path.name} is damaged at line {verdict.broken_at} "
                                  f"({verdict.reason}); move it aside to start a new log")
            base = self._base()
            row: dict[str, Any] = {**body, "v": VERSION, "seq": base + len(rows) + 1,
                                   "prev": verdict.head, "ts": round(self.clock(), 3)}
            row["hash"] = _digest(verdict.head, row)
            self._write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            return row

    def _base(self) -> int:
        lines, _ = self._lines()
        if lines and b'"base"' in lines[0]:
            first = json.loads(lines[0])
            return int(first["base"]["seq"]) if "base" in first else 0
        return 0

    def _write(self, text: str) -> None:
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, text.encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    def _cut_torn(self) -> None:
        _, torn = self._lines()
        if not torn:
            return
        with self.path.open("rb+") as handle:
            handle.truncate(handle.seek(0, os.SEEK_END) - len(torn))
        with self.path.with_name(self.path.name + ".torn").open("ab") as kept:
            kept.write(torn + b"\n")

    def prune_prefix(self, drop: Callable[[dict[str, Any]], bool]) -> int:
        """Remove the oldest rows while ``drop`` says so; the chain continues from the last
        removed row. Returns how many went."""
        with held(self.guard):
            self._cut_torn()
            rows, verdict = self._walk()
            if not verdict.ok:
                raise ChainBroken(f"{self.path.name} is damaged at line {verdict.broken_at}")
            cut = 0
            while cut < len(rows) and drop(rows[cut]):
                cut += 1
            if not cut:
                return 0
            keep = rows[cut:]
            first = {"base": {"seq": rows[cut - 1]["seq"], "hash": rows[cut - 1]["hash"]}}
            text = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                           for r in [first, *keep])
            write_text(self.path, text)
            self.path.chmod(0o600)
            return cut
