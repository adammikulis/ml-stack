"""One quarantine store for every kind of subject, with one state machine.

A subject is ``(kind, key)``. It is clear, watched, quarantined or released; only a person
releases or purges. Held text lives in files beside the sealed state, redacted and capped,
and is read back only with a `HumanGrant`.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.files import write_text
from ml_stack.lock import only_one
from ml_stack.sentinel import moves
from ml_stack.sentinel.events import Bus, Event, Severity
from ml_stack.sentinel.human import HumanGrant
from ml_stack.sentinel.redaction import redact, redact_value
from ml_stack.sentinel.sealed import SealedFile

__all__ = ["KINDS", "TRANSITIONS", "Holding", "Limits", "Record", "State", "Store", "TransitionRefused",
           "fingerprint", "placeholder", "sentinel_dir"]

KINDS = ("message", "tool_call", "tool", "session", "memory", "model", "artifact",
         "binary", "config", "mcp_server", "credential", "peer", "server", "caller")
FILE_KINDS = ("model", "artifact", "binary", "config")
TEXT_KINDS = ("message", "tool_call", "memory", "session")


class State(StrEnum):
    CLEAR = "clear"
    WATCH = "watch"
    QUARANTINED = "quarantined"
    RELEASED = "released"


TRANSITIONS: dict[State, frozenset[State]] = {
    State.CLEAR: frozenset({State.WATCH, State.QUARANTINED}),
    State.WATCH: frozenset({State.WATCH, State.CLEAR, State.QUARANTINED}),
    State.QUARANTINED: frozenset({State.RELEASED}),
    State.RELEASED: frozenset({State.WATCH, State.QUARANTINED}),
}
"""Which states a subject may move to from each state."""


class TransitionRefused(ValueError):
    """The state machine does not allow the move that was asked for."""


@dataclass(frozen=True, slots=True)
class Holding:
    """What a quarantine keeps: ``text`` held as an inert copy, or the file at ``path`` moved
    aside (``move="link"`` moves a symlink itself)."""

    text: str | None = None
    path: Path | str | None = None
    move: str = "file"


@dataclass(frozen=True, slots=True)
class Limits:
    """What keeps the store from growing without bound."""

    max_records: int = 4000
    max_payload_bytes: int = 64 * 1024
    max_payload_total: int = 16 * 1024 * 1024
    clear_after_s: float = 7 * 86400.0
    released_payload_s: float = 30 * 86400.0


@dataclass
class Record:
    id: str
    kind: str
    key: str
    state: State
    reason: str
    evidence: dict[str, Any]
    created: float
    updated: float
    history: list[dict[str, Any]] = field(default_factory=list)
    fingerprint: str = ""
    held: dict[str, Any] | None = None
    action: dict[str, Any] | None = None
    purged: bool = False

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "key": self.key, "state": self.state.value,
                "reason": self.reason, "evidence": self.evidence, "created": self.created,
                "updated": self.updated, "history": self.history,
                "fingerprint": self.fingerprint, "held": self.held, "action": self.action,
                "purged": self.purged}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Record:
        return cls(id=data["id"], kind=data["kind"], key=data["key"],
                   state=State(data["state"]), reason=data["reason"],
                   evidence=data["evidence"], created=data["created"], updated=data["updated"],
                   history=data["history"], fingerprint=data.get("fingerprint", ""),
                   held=data.get("held"), action=data.get("action"),
                   purged=data.get("purged", False))


_SPACE = re.compile(r"\s+")


def fingerprint(text: str) -> str:
    """A digest of ``text`` with case and runs of whitespace normalised."""
    return hashlib.sha256(_SPACE.sub(" ", text.strip().lower()).encode()).hexdigest()


def placeholder(ident: str) -> str:
    """What a model is shown in place of held content."""
    return f"[content withheld by sentinel, id {ident}]"


def sentinel_dir() -> Path:
    """Where sentinel keeps its state, key, log and held text."""
    return home.state("sentinel")


Hook = Callable[[Record], None]


class Store:
    """Records of every subject sentinel has watched or quarantined."""

    def __init__(self, root: Path | None = None, bus: Bus | None = None, *,
                 limits: Limits | None = None, roots: list[Path] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.root = Path(root) if root is not None else sentinel_dir()
        self.bus = bus or Bus()
        self.limits = limits or Limits()
        self.roots = roots
        self.clock = clock
        self._file = SealedFile(self.root / "state.json")
        self._lock = threading.RLock()
        self.on_quarantine: dict[str, list[Hook]] = {}
        self.on_release: dict[str, list[Hook]] = {}
        self._records: dict[str, Record] = {}
        self._counters: dict[str, int] = {"seq": 0, "overflow": 0}
        self.tampered = False
        self._load()

    def _load(self) -> None:
        loaded = self._file.load()
        if loaded.status in ("recovered", "tampered"):
            self.tampered = loaded.status == "tampered"
            self._emit("sentinel.tamper", Severity.CRITICAL, "",
                       {"state": loaded.status})
        data = loaded.payload
        self._records = {r["id"]: Record.from_json(r) for r in data.get("records", [])}
        self._counters.update(data.get("counters", {}))
        if loaded.status == "tampered":
            self._counters["tampered"] = 1
        self.tampered = self.tampered or bool(self._counters.get("tampered"))

    def _emit(self, kind: str, severity: Severity, subject: str, evidence: dict[str, Any],
              ) -> None:
        self.bus.emit(Event(kind, severity, "store", subject, evidence, self.clock()))

    def _refresh(self) -> None:
        data = self._file.load().payload
        if data:
            self._records = {r["id"]: Record.from_json(r) for r in data.get("records", [])}
            self._counters.update(data.get("counters", {}))

    def _save(self) -> None:
        self._file.save({"records": [r.to_json() for r in self._records.values()],
                         "counters": self._counters})

    def _locked(self):
        return only_one(self.root / "state.lock")

    # -- reading ---------------------------------------------------------------------
    def get(self, ident: str) -> Record | None:
        with self._lock:
            self._refresh()
            return self._records.get(ident)

    def find(self, kind: str, key: str) -> Record | None:
        with self._lock:
            self._refresh()
            return next((r for r in self._records.values()
                         if r.kind == kind and r.key == key), None)

    def records(self, *, kind: str = "", state: State | None = None) -> list[Record]:
        with self._lock:
            self._refresh()
            return [r for r in self._records.values()
                    if (not kind or r.kind == kind) and (state is None or r.state == state)]

    def state_of(self, kind: str, key: str) -> State:
        record = self.find(kind, key)
        return record.state if record else State.CLEAR

    def blocked(self, kind: str, key: str) -> bool:
        """Whether a subject is quarantined now. Everything is blocked when the sealed
        state failed its seal and no copy held."""
        return self.tampered or self.state_of(kind, key) == State.QUARANTINED

    def find_fingerprint(self, digest: str) -> Record | None:
        """The quarantined record whose held text has this fingerprint, if any."""
        with self._lock:
            self._refresh()
            return next((r for r in self._records.values()
                         if r.fingerprint == digest and r.state == State.QUARANTINED), None)

    # -- moving between states -------------------------------------------------------
    def _new(self, kind: str, key: str, reason: str, evidence: dict[str, Any]) -> Record | None:
        if kind not in KINDS:
            raise ValueError(f"unknown subject kind {kind!r}")
        self._prune()
        if len(self._records) >= 2 * self.limits.max_records:
            self._counters["overflow"] += 1
            return None
        self._counters["seq"] += 1
        ident = "q-" + hashlib.sha256(
            f"{kind}|{key}|{self._counters['seq']}".encode()).hexdigest()[:10]
        now = self.clock()
        return Record(ident, kind, key, State.CLEAR, "", redact_value(evidence), now, now)

    def _move(self, record: Record, to: State, reason: str, actor: str,
              evidence: dict[str, Any]) -> None:
        if to not in TRANSITIONS[record.state]:
            raise TransitionRefused(f"{record.state.value} -> {to.value} is not allowed")
        record.history.append({"ts": self.clock(), "from": record.state.value, "to": to.value,
                               "actor": actor, "reason": redact(reason)})
        del record.history[:-50]
        record.state, record.reason, record.updated = to, redact(reason), self.clock()
        record.evidence = {**record.evidence, **redact_value(evidence)}
        severity = Severity.CRITICAL if to == State.QUARANTINED else Severity.NOTICE
        self._emit(f"quarantine.{to.value}", severity, f"{record.kind}:{record.key}",
                   {"id": record.id, "actor": actor, "reason": redact(reason)})

    def watch(self, kind: str, key: str, reason: str, evidence: dict[str, Any] | None = None,
              ) -> Record | None:
        """Put a subject under watch (or keep it there)."""
        with self._lock, self._locked():
            self._refresh()
            record = self._find_unlocked(kind, key) or self._new(kind, key, reason, evidence or {})
            if record is None:
                self._save()
                return None
            if record.state == State.QUARANTINED:
                return record
            self._records[record.id] = record
            if record.state == State.WATCH:
                record.updated = self.clock()
                record.evidence = {**record.evidence, **redact_value(evidence or {})}
                self._save()
                return record
            self._move(record, State.WATCH, reason, "sentinel", evidence or {})
            self._save()
            return record

    def _find_unlocked(self, kind: str, key: str) -> Record | None:
        return next((r for r in self._records.values()
                     if r.kind == kind and r.key == key), None)

    def clear(self, kind: str, key: str, reason: str = "no further signal") -> Record | None:
        """Return a watched subject to clear."""
        with self._lock, self._locked():
            self._refresh()
            record = self._find_unlocked(kind, key)
            if record is None or record.state != State.WATCH:
                return record
            self._move(record, State.CLEAR, reason, "sentinel", {})
            self._save()
            return record

    def quarantine(self, subject: tuple[str, str], reason: str,
                   evidence: dict[str, Any] | None = None, holding: Holding | None = None,
                   actor: str = "sentinel") -> Record | None:
        """Quarantine a subject: hold its text or move its file aside, then run the hooks
        for its kind. Returns None only when the store is full."""
        kind, key = subject
        with self._lock, self._locked():
            self._refresh()
            record = self._find_unlocked(kind, key)
            if record is not None and record.state == State.QUARANTINED:
                return record
            record = record or self._new(kind, key, reason, evidence or {})
            if record is None:
                self._save()
                self._emit("store.overflow", Severity.WARNING, f"{kind}:{key}",
                           {"overflow": self._counters["overflow"]})
                return None
            self._records[record.id] = record
            self._apply(record, holding or Holding())
            self._move(record, State.QUARANTINED, reason, actor, evidence or {})
            self._save()
        self._run(self.on_quarantine, record)
        return record

    def _apply(self, record: Record, holding: Holding) -> None:
        if holding.text is not None:
            self._hold_text(record, holding.text)
        if holding.path is not None and record.kind in FILE_KINDS:
            try:
                record.action = moves.hold(holding.path, record.id, move=holding.move,
                                           roots=self.roots)
            except moves.MoveRefused as exc:
                record.evidence = {**record.evidence, "move_refused": redact(str(exc))}
                self._emit("quarantine.move_refused", Severity.WARNING,
                           f"{record.kind}:{record.key}", {"id": record.id, "why": str(exc)})

    def _hold_text(self, record: Record, text: str) -> None:
        clean = redact(text)
        record.fingerprint = fingerprint(text)
        room = self.limits.max_payload_total - self._payload_total()
        data = clean.encode("utf-8", errors="replace")[:self.limits.max_payload_bytes]
        if len(data) > room:
            record.held = {"dropped": "the store's text limit is reached"}
            return
        body = data.decode("utf-8", errors="ignore")
        name = f"{record.id}.json"
        write_text(self.root / "items" / name,
                   json.dumps({"id": record.id, "text": body, "redacted": True}))
        (self.root / "items" / name).chmod(0o600)
        record.held = {"file": name, "bytes": len(data),
                       "truncated": len(clean.encode()) > len(data),
                       "sha256": hashlib.sha256(
                           (self.root / "items" / name).read_bytes()).hexdigest()}

    def _payload_total(self) -> int:
        return sum(r.held.get("bytes", 0) for r in self._records.values()
                   if r.held and not r.purged)

    # -- the human's side ------------------------------------------------------------
    def release(self, ident: str, grant: HumanGrant, reason: str = "released by a person",
                ) -> Record:
        """Release a quarantined subject: restore its file, run the release hooks. A
        grant for ``release`` on this id is required."""
        grant.check("release", ident)
        if self.tampered:
            raise TransitionRefused("the sealed state failed its check; nothing is released")
        with self._lock, self._locked():
            self._refresh()
            record = self._records.get(ident)
            if record is None:
                raise KeyError(ident)
            if record.state != State.QUARANTINED:
                raise TransitionRefused(f"{ident} is {record.state.value}, not quarantined")
            if record.action and not record.purged:
                moves.restore(record.action)
            self._move(record, State.RELEASED, reason, "human", {})
            self._save()
        self._run(self.on_release, record)
        return record

    def read(self, ident: str, grant: HumanGrant) -> str:
        """The redacted text held for ``ident``. A grant for ``inspect`` on this id is
        required."""
        grant.check("inspect", ident)
        record = self.get(ident)
        if record is None or not record.held or "file" not in record.held or record.purged:
            raise KeyError(ident)
        raw = (self.root / "items" / record.held["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record.held["sha256"]:
            self._emit("sentinel.tamper", Severity.CRITICAL, f"{record.kind}:{record.key}",
                       {"id": ident, "what": "held text changed on disk"})
            raise TransitionRefused("the held text changed on disk")
        return str(json.loads(raw)["text"])

    def purge(self, ident: str, grant: HumanGrant) -> Record:
        """Delete what is held for ``ident`` (its text or its moved file) and keep the
        record. A grant for ``purge`` on this id is required."""
        grant.check("purge", ident)
        with self._lock, self._locked():
            self._refresh()
            record = self._records.get(ident)
            if record is None:
                raise KeyError(ident)
            if record.state == State.QUARANTINED and record.action:
                moves.purge(record.action)
            if record.held and "file" in record.held:
                (self.root / "items" / record.held["file"]).unlink(missing_ok=True)
            record.purged, record.held = True, None
            record.history.append({"ts": self.clock(), "from": record.state.value,
                                   "to": "purged", "actor": "human", "reason": "purged"})
            self._emit("quarantine.purged", Severity.NOTICE, f"{record.kind}:{record.key}",
                       {"id": ident})
            self._save()
            return record

    def _run(self, hooks: dict[str, list[Hook]], record: Record) -> None:
        for hook in hooks.get(record.kind, []):
            try:
                hook(record)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                self._emit("effect.failed", Severity.WARNING, f"{record.kind}:{record.key}",
                           {"id": record.id, "why": redact(str(exc))})

    # -- bounds ----------------------------------------------------------------------
    def _prune(self) -> None:
        now = self.clock()
        for record in list(self._records.values()):
            stale_clear = (record.state == State.CLEAR
                           and now - record.updated > self.limits.clear_after_s)
            if stale_clear:
                del self._records[record.id]
            elif (record.state == State.RELEASED and record.held and "file" in record.held
                  and now - record.updated > self.limits.released_payload_s):
                (self.root / "items" / record.held["file"]).unlink(missing_ok=True)
                record.held = None
        excess = len(self._records) - self.limits.max_records
        if excess > 0:
            idle = sorted((r for r in self._records.values()
                           if r.state in (State.CLEAR, State.WATCH, State.RELEASED)),
                          key=lambda r: r.updated)
            for record in idle[:excess]:
                if record.held and "file" in record.held:
                    (self.root / "items" / record.held["file"]).unlink(missing_ok=True)
                del self._records[record.id]
