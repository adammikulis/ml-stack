"""An append-only JSON-lines log whose rows are numbered and hash-chained."""

from __future__ import annotations

import hashlib
import json
import os
import threading
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
REVERIFY_S = 30.0


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
def held(path: Path, *, shared: bool = False) -> Iterator[None]:
    """Hold ``path`` exclusively for the block, polling every few milliseconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        while not (lock.take(fd, shared=True) if shared else lock.take(fd)):
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


class _Seen:
    """The rows one process has verified up to byte ``offset`` and the file's identity then."""

    __slots__ = ("at", "base", "ino", "lines", "mtime_ns", "offset", "prev", "rows")

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.prev, self.base, self.lines, self.offset = GENESIS, 0, 0, 0
        self.ino, self.mtime_ns, self.at = 0, 0, time.monotonic()

    def take(self, line: bytes) -> Verdict | None:
        """Verify one line onto the rows; a Verdict when it does not follow."""
        number = self.lines + 1
        try:
            row = json.loads(line)
        except ValueError:
            return Verdict(False, len(self.rows), self.prev, number, "row is not JSON")
        if not isinstance(row, dict):
            return Verdict(False, len(self.rows), self.prev, number, "row is not an object")
        if number == 1 and "base" in row:
            self.prev, self.base = str(row["base"]["hash"]), int(row["base"]["seq"])
        else:
            if row.get("prev") != self.prev or row.get("hash") != _digest(self.prev, row):
                return Verdict(False, len(self.rows), self.prev, number, "hash does not follow")
            if row.get("seq") != self.base + len(self.rows) + 1:
                return Verdict(False, len(self.rows), self.prev, number, "sequence number skips")
            self.rows.append(row)
            self.prev = row["hash"]
        self.lines = number
        return None


class ChainLog:
    """Rows with ``seq``, ``prev`` and ``hash``, fsynced on append.

    A final line without a newline (a crash mid-write) is cut off the next time the log is
    appended to, and the cut bytes are kept in ``<name>.torn``. A process keeps the rows it has
    verified and reads only the bytes added since. `verify` and `prune_prefix` walk the whole
    file; so does an append when the file is not as the last appender left it (``<name>.stamp``),
    and any read after ``REVERIFY_S``.
    """

    def __init__(self, path: Path | str, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.clock = clock
        self.guard = self.path.with_name(self.path.name + ".lock")
        self._stamp = self.path.with_name(self.path.name + ".stamp")
        self._seen: _Seen | None = None
        self._mutex = threading.RLock()

    def _sync(self, fresh: bool = False) -> tuple[_Seen, Verdict]:
        """Bring the verified rows up to the file's present state; ``fresh`` walks all of it."""
        with self._mutex:
            try:
                st = self.path.stat()
            except FileNotFoundError:
                self._seen = None
                return _Seen(), Verdict(True, 0, GENESIS)
            seen = self._seen
            if (fresh or seen is None or seen.ino != st.st_ino or st.st_size < seen.offset
                    or (st.st_size == seen.offset and st.st_mtime_ns != seen.mtime_ns)
                    or time.monotonic() - seen.at >= REVERIFY_S):
                seen = _Seen()
            bad = self._advance(seen, st.st_size) if st.st_size > seen.offset else None
            seen.ino, seen.mtime_ns = st.st_ino, st.st_mtime_ns
            self._seen = seen
            return seen, bad or Verdict(True, len(seen.rows), seen.prev)

    def _advance(self, seen: _Seen, size: int) -> Verdict | None:
        """Verify the whole lines between ``seen.offset`` and ``size``."""
        try:
            with self.path.open("rb") as handle:
                handle.seek(seen.offset)
                data = handle.read(size - seen.offset)
        except FileNotFoundError:
            return None
        pos = 0
        while (nl := data.find(b"\n", pos)) >= 0:
            line = data[pos:nl]
            if line.strip() and (bad := seen.take(line)) is not None:
                return bad
            seen.offset += nl + 1 - pos
            pos = nl + 1
        return None

    def rows(self) -> list[dict[str, Any]]:
        """The rows whose chain holds, in order."""
        return list(self._sync()[0].rows)

    def after(self, seq: int) -> list[dict[str, Any]]:
        """The verified rows numbered above ``seq``."""
        seen = self._sync()[0]
        return seen.rows[max(0, seq - seen.base):]

    def verify(self, anchor: str = "") -> Verdict:
        """Whether the chain holds, and when ``anchor`` is a head hash, that it is still in it."""
        seen, verdict = self._sync(fresh=True)
        if verdict.ok and anchor and anchor not in {GENESIS, *(r["hash"] for r in seen.rows)}:
            return Verdict(False, verdict.rows, verdict.head, 0, "anchor is not in the log")
        return verdict

    def head(self) -> str:
        """The hash of the last row."""
        return self._sync()[1].head

    def append(self, body: dict[str, Any]) -> dict[str, Any]:
        """Add a row made of ``body``; returns it with its ``seq``, ``prev`` and ``hash``."""
        with held(self.guard):
            self._cut_torn()
            seen, verdict = self._sync(fresh=not self._stamped())
            if not verdict.ok:
                raise ChainBroken(f"{self.path.name} is damaged at line {verdict.broken_at} "
                                  f"({verdict.reason}); move it aside to start a new log")
            row: dict[str, Any] = {**body, "v": VERSION, "seq": seen.base + len(seen.rows) + 1,
                                   "prev": verdict.head, "ts": round(self.clock(), 3)}
            row["hash"] = _digest(verdict.head, row)
            self._write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            return row

    def extend(self, rows: list[dict[str, Any]]) -> int:
        """Append rows another log produced, verbatim, each following the one before it."""
        with held(self.guard):
            self._cut_torn()
            seen, verdict = self._sync(fresh=not self._stamped())
            if not verdict.ok:
                raise ChainBroken(f"{self.path.name} is damaged at line {verdict.broken_at}")
            prev, seq = verdict.head, seen.base + len(seen.rows)
            for row in rows:
                if row.get("prev") != prev or row.get("seq") != seq + 1 or row.get("hash") != _digest(prev, row):
                    raise ChainBroken(f"row {seq + 1} does not follow {self.path.name}")
                prev, seq = row["hash"], seq + 1
            if rows:
                self._write("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows))
            return len(rows)

    def _write(self, text: str) -> None:
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, text.encode())
            os.fsync(fd)
            st = os.fstat(fd)
            self._stamp.write_text(f"{st.st_ino} {st.st_size} {st.st_mtime_ns}")
        finally:
            os.close(fd)

    def _stamped(self) -> bool:
        """Whether the file is exactly as the last appender left it, by inode, size and mtime."""
        try:
            st = self.path.stat()
            return self._stamp.read_text() == f"{st.st_ino} {st.st_size} {st.st_mtime_ns}"
        except OSError:
            return False

    def _torn(self) -> bytes:
        """The bytes after the last newline, read back from the end of the file."""
        try:
            with self.path.open("rb") as handle:
                pos = handle.seek(0, os.SEEK_END)
                tail = b""
                while pos > 0:
                    step = min(4096, pos)
                    pos -= step
                    handle.seek(pos)
                    tail = handle.read(step) + tail
                    if (cut := tail.rfind(b"\n")) >= 0:
                        return tail[cut + 1:]
                return tail
        except FileNotFoundError:
            return b""

    def _cut_torn(self) -> None:
        torn = self._torn()
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
            seen, verdict = self._sync(fresh=True)
            rows = seen.rows
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
