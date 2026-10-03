"""The facts, in one sealed file in ml-stack's state directory."""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import home, lock
from ml_stack.memory.facts import BUILD_BOUND, KINDS, SOURCES, STALE_DAYS, Fact, Refused, check
from ml_stack.sentinel.human import protect
from ml_stack.sentinel.sealed import SealedFile

__all__ = ["MAX_FACTS", "SCHEMA_VERSION", "Store", "Tampered", "current_scope", "stale_reason"]

SCHEMA_VERSION = 1
MAX_FACTS = 300
DAY = 86400.0


class Tampered(RuntimeError):
    """The store file failed its seal and no earlier copy held."""


def current_scope() -> dict[str, str]:
    """The machine and the managed llama.cpp build in use now."""
    from ml_stack.serve.binary import managed_current

    link = managed_current()
    build = link.resolve().name if link.exists() else ""
    return {"machine": platform.node(), "build": build}


def stale_reason(fact: Fact, now: float, scope: dict[str, str]) -> str:
    """Why ``fact`` should be re-checked before it is relied on, or an empty string."""
    if fact.kind not in BUILD_BOUND:
        return ""
    then, here = fact.scope.get("build", ""), scope.get("build", "")
    if then and here and then != here:
        return f"learned on build {then}; build {here} is in use"
    days = (now - fact.last_confirmed) / DAY
    if days > STALE_DAYS[fact.kind]:
        return f"last confirmed {days:.0f} days ago"
    return ""


class Store:
    """Facts sealed on disk: the file is mode 0600, written atomically, and a read that
    fails its seal falls back to the previous copy or refuses."""

    def __init__(self, path: Path | None = None, *, clock: Callable[[], float] = time.time,
                 scope: Callable[[], dict[str, str]] = current_scope) -> None:
        self.path = Path(path) if path else home.state("memory", "facts.json")
        self.clock, self.scope = clock, scope
        self._file = SealedFile(self.path)
        protect(self.path.parent)

    # -- reading --------------------------------------------------------------------
    def _load(self) -> tuple[dict[str, Any], str]:
        loaded = self._file.load()
        if loaded.status == "tampered":
            return {}, "tampered"
        payload = loaded.payload
        if payload and payload.get("schema_version") != SCHEMA_VERSION:
            return {}, "tampered"
        return payload, loaded.status

    @property
    def status(self) -> str:
        """``fresh``, ``ok``, ``recovered`` (the previous copy was used) or ``tampered``."""
        return self._load()[1]

    def facts(self) -> list[Fact]:
        """Every fact, oldest first; empty when the store is tampered with."""
        payload, _ = self._load()
        return [Fact.from_json(row) for row in payload.get("facts", [])]

    def get(self, ident: str) -> Fact | None:
        return next((f for f in self.facts() if f.id == ident), None)

    def stale(self, fact: Fact) -> str:
        """Why ``fact`` needs re-checking now, or an empty string."""
        return stale_reason(fact, self.clock(), self.scope())

    def stats(self) -> dict[str, Any]:
        facts = self.facts()
        return {"status": self.status, "path": str(self.path), "facts": len(facts),
                "limit": MAX_FACTS, "by_kind": {k: sum(f.kind == k for f in facts) for k in KINDS},
                "stale": sum(bool(self.stale(f)) for f in facts)}

    def export(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "facts": [f.to_json() for f in self.facts()]}

    # -- writing --------------------------------------------------------------------
    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        self._file.save({"schema_version": SCHEMA_VERSION, **payload})
        for each in (self.path, self._file.prev, self._file.keyfile):
            if each.exists():
                os.chmod(each, 0o600)

    def _edit(self, change: Callable[[dict[str, Any], list[dict[str, Any]]], Any]) -> Any:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with lock.only_one(self.path.parent / "store.lock", timeout=10, announce=lambda _: None):
            payload, status = self._load()
            if status == "tampered":
                raise Tampered(f"{self.path} failed its integrity check; "
                               "`ml-stack-memory forget --all` starts a new store")
            rows = list(payload.get("facts", []))
            out = change(payload, rows)
            self._write({"next": payload.get("next", 1), "facts": rows})
            return out

    def add(self, text: str, kind: str = "note", source: str = "user-said", *, model: str = "",
            person: bool = False) -> Fact:
        """Store a fact, or confirm the one with the same text. ``person`` is true when the
        person typed ``text`` themselves. Raises ``Refused`` or ``Tampered``."""
        if kind not in KINDS:
            raise Refused(f"kind is one of {', '.join(KINDS)}")
        if source not in SOURCES:
            raise Refused(f"source is one of {', '.join(SOURCES)}")
        text = check(text, person=person)
        now, scope = self.clock(), dict(self.scope())
        if model:
            scope["model"] = check(model, person=True)[:80]

        def change(payload: dict[str, Any], rows: list[dict[str, Any]]) -> Fact:
            for row in rows:
                if str(row.get("text", "")).casefold() == text.casefold() and row.get("kind") == kind:
                    row["last_confirmed"] = now
                    row["confirm_count"] = int(row.get("confirm_count", 1)) + 1
                    row["scope"] = {**dict(row.get("scope") or {}), **scope}
                    return Fact.from_json(row)
            if len(rows) >= MAX_FACTS:
                raise Refused(f"the store holds {MAX_FACTS} facts; forget some first "
                              "(ml-stack-memory list, forget ID)")
            number = int(payload.get("next", 1))
            payload["next"] = number + 1
            fact = Fact(f"m{number:04d}", text, kind, source, now, now, 1, scope)
            rows.append(fact.to_json())
            return fact

        return self._edit(change)

    def confirm(self, ident: str) -> Fact:
        """Mark a fact as checked now, under the build in use now."""
        now, scope = self.clock(), dict(self.scope())

        def change(payload: dict[str, Any], rows: list[dict[str, Any]]) -> Fact:
            for row in rows:
                if row.get("id") == ident:
                    row["last_confirmed"] = now
                    row["confirm_count"] = int(row.get("confirm_count", 1)) + 1
                    row["scope"] = {**dict(row.get("scope") or {}), **scope}
                    return Fact.from_json(row)
            raise KeyError(ident)

        return self._edit(change)

    def forget(self, ident: str) -> bool:
        """Delete one fact; true when it existed."""
        def change(payload: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
            kept = [r for r in rows if r.get("id") != ident]
            found = len(kept) != len(rows)
            rows[:] = kept
            return found

        return self._edit(change)

    def forget_all(self) -> int:
        """Delete every fact and the seal with them, so a tampered store starts again;
        returns how many facts were held."""
        held = len(self.facts())
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with lock.only_one(self.path.parent / "store.lock", timeout=10, announce=lambda _: None):
            for each in (self.path, self._file.prev, self._file.keyfile):
                each.unlink(missing_ok=True)
        return held
