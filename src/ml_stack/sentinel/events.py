"""The security event: one typed record, a bounded append-only log, and subscribers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, NamedTuple

from ml_stack.lock import only_one
from ml_stack.sentinel.redaction import redact_value
from ml_stack.sentinel.sealed import SealedFile

__all__ = ["Bus", "Event", "EventLog", "Severity", "Verified"]

logger = logging.getLogger("ml_stack.sentinel")
logger.addHandler(logging.NullHandler())

VERSION = 1


class Severity(IntEnum):
    INFO = 0
    NOTICE = 1
    WARNING = 2
    CRITICAL = 3

    @classmethod
    def parse(cls, name: str) -> Severity:
        return cls[name.strip().upper()]


@dataclass(frozen=True, slots=True)
class Event:
    """What happened (``kind``), how bad (``severity``), who saw it (``source``), to what
    (``subject``, as ``kind:key``), and references that back it up (``evidence``)."""

    kind: str
    severity: Severity
    source: str
    subject: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        """The redacted JSON form written to the log."""
        return {"version": VERSION, "ts": round(self.ts, 3), "kind": self.kind,
                "severity": self.severity.name.lower(), "source": self.source,
                "subject": redact_value(self.subject),
                "evidence": redact_value(dict(self.evidence))}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Event:
        return cls(kind=str(record["kind"]), severity=Severity.parse(str(record["severity"])),
                   source=str(record.get("source", "")), subject=str(record.get("subject", "")),
                   evidence=dict(record.get("evidence", {})), ts=float(record.get("ts", 0.0)))


class Chain(NamedTuple):
    """Where a hash chain stands: records written, the last hash, and the hash that precedes
    the oldest record still on disk."""

    count: int
    last: str
    base: str


GENESIS = "0" * 64


@dataclass(frozen=True, slots=True)
class Verified:
    """The outcome of checking a chained log: whether it holds, what is wrong, and its head."""

    ok: bool
    problems: tuple[str, ...]
    records: int
    head: str


def _link(prev: str, record: Mapping[str, Any]) -> str:
    body = json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256((prev + body).encode()).hexdigest()


class EventLog:
    """JSON lines on disk, mode 0600, each carrying the hash of the line before it. The
    head (count and last hash) is kept in a sealed file beside the log, so a line that is
    edited, removed, reordered or cut from the end is found by `verify`. The log is rotated
    when a file passes ``max_bytes`` and kept to ``keep`` files."""

    def __init__(self, path: Path, *, max_bytes: int = 1_000_000, keep: int = 4,
                 anchor: Path | None = None) -> None:
        self.path, self.max_bytes, self.keep = Path(path), max_bytes, keep
        self.anchor = anchor
        self._head = SealedFile(self.path.with_name(self.path.name + ".head"))
        self._lock = threading.Lock()

    def _chain(self) -> Chain | None:
        loaded = self._head.load()
        if loaded.status == "fresh":
            return Chain(0, GENESIS, GENESIS)
        if loaded.status != "ok" and loaded.status != "recovered":
            return None
        data = loaded.payload
        return Chain(int(data["count"]), str(data["last"]), str(data["base"]))

    def append(self, event: Event) -> None:
        """Add ``event`` to the chain. A head that fails its seal starts a new chain whose
        first record says so."""
        record = event.to_record()
        with self._lock, only_one(self.path.with_name(self.path.name + ".lock")):
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            chain = self._chain()
            if chain is None:
                chain = Chain(0, GENESIS, GENESIS)
                record["evidence"] = {**record["evidence"], "chain": "head failed its seal"}
            if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                chain = self._rotate(chain)
            record["seq"], record["prev"] = chain.count, chain.last
            record["hash"] = _link(chain.last, {k: v for k, v in record.items() if k != "hash"})
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                line = json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n"
                os.write(fd, line.encode())
            finally:
                os.close(fd)
            self._head.save({"count": chain.count + 1, "last": record["hash"],
                             "base": chain.base})
            if self.anchor is not None:
                self._write_anchor(chain.count + 1, record["hash"])

    def _write_anchor(self, count: int, last: str) -> None:
        if self.anchor is None:
            return
        self.anchor.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.anchor, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, f"{count} {last}\n".encode())
        finally:
            os.close(fd)

    def _numbered(self, number: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{number}")

    def _rotate(self, chain: Chain) -> Chain:
        oldest = self._numbered(self.keep - 1) if self.keep > 1 else self.path
        base = chain.base
        if oldest.exists():
            lines = oldest.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines:
                try:
                    base = str(json.loads(lines[-1])["hash"])
                except (ValueError, KeyError, TypeError):
                    base = chain.base
            oldest.unlink()
        for number in range(self.keep - 2, 0, -1):
            if self._numbered(number).exists():
                self._numbered(number).rename(self._numbered(number + 1))
        if self.keep > 1:
            self.path.rename(self._numbered(1))
        return Chain(chain.count, chain.last, base)

    def files(self) -> list[Path]:
        """The log files, oldest first."""
        older = [self._numbered(n) for n in range(self.keep - 1, 0, -1)]
        return [p for p in (*older, self.path) if p.exists()]

    def read(self) -> Iterator[Event]:
        """Every readable event, oldest first; a damaged line is skipped."""
        for file in self.files():
            for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    yield Event.from_record(json.loads(line))
                except (ValueError, KeyError, TypeError):
                    continue

    def head(self) -> str:
        """``count hash``: the line to write down somewhere an attacker cannot reach."""
        chain = self._chain()
        return "unreadable" if chain is None else f"{chain.count} {chain.last}"

    def verify(self, anchor: str = "") -> Verified:
        """Check every link, the sequence, the sealed head, and ``anchor`` (a ``head()``
        line kept elsewhere) when given."""
        problems: list[str] = []
        chain = self._chain()
        if chain is None:
            problems.append("the head file failed its seal")
            chain = Chain(0, GENESIS, GENESIS)
        prev, seen, first = chain.base, 0, True
        expected = -1
        for file in self.files():
            for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
                where = f"{file.name}:{number}"
                try:
                    record = json.loads(line)
                    claimed = str(record.pop("hash"))
                except (ValueError, KeyError, AttributeError):
                    problems.append(f"{where}: not a chained record")
                    continue
                if record.get("prev") != prev:
                    problems.append(f"{where}: does not follow the record before it")
                if claimed != _link(str(record.get("prev", "")), record):
                    problems.append(f"{where}: edited")
                seq = record.get("seq", -1)
                if not first and seq != expected:
                    problems.append(f"{where}: sequence {seq}, expected {expected}")
                first, prev, seen = False, claimed, seen + 1
                expected = seq + 1 if isinstance(seq, int) else -1
        if prev != chain.last:
            problems.append("the log ends before the sealed head: records were cut off")
        if chain.count and not seen:
            problems.append("the log is empty but the head says it was not")
        if anchor and anchor.strip() != f"{chain.count} {chain.last}":
            problems.append("the head differs from the anchor")
        return Verified(not problems, tuple(problems), seen, f"{chain.count} {chain.last}")


class Bus:
    """Fans an event out to the log and to subscribers. A subscriber that raises is
    logged and does not stop the others."""

    def __init__(self, log: EventLog | None = None, *, recent: int = 2000) -> None:
        self.log = log
        self._subscribers: list[Callable[[Event], None]] = []
        self._recent: list[Event] = []
        self._most = recent
        self._lock = threading.Lock()

    def subscribe(self, fn: Callable[[Event], None]) -> Callable[[], None]:
        """Call ``fn`` with every event from now on; returns the call that unsubscribes."""
        with self._lock:
            self._subscribers.append(fn)

        def stop() -> None:
            with self._lock:
                if fn in self._subscribers:
                    self._subscribers.remove(fn)
        return stop

    def emit(self, event: Event) -> Event:
        """Record ``event`` and hand it to every subscriber."""
        with self._lock:
            self._recent.append(event)
            del self._recent[:-self._most]
            listeners = list(self._subscribers)
        if self.log is not None:
            try:
                self.log.append(event)
            except OSError as exc:
                logger.warning("sentinel log not written: %s", exc)
        for fn in listeners:
            try:
                fn(event)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
                logger.exception("sentinel subscriber failed")
        return event

    def recent(self, *, kind: str = "", at_least: Severity = Severity.INFO,
               since: float = 0.0) -> list[Event]:
        """Events held in memory, oldest first, whose kind starts with ``kind``."""
        with self._lock:
            return [e for e in self._recent
                    if e.severity >= at_least and e.ts >= since
                    and (not kind or e.kind.startswith(kind))]
