"""Who owns which branch, worktree, port, file or server."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ml_stack.files import read_json, write_json
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import AGENT, Denied, Identity

__all__ = ["KINDS", "Claims", "Conflict", "alive", "normal"]

KINDS = ("branch", "worktree", "port", "file", "server")
PATH_KINDS = ("worktree", "file")
WORD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,199}$")
VERSION = 1


class Conflict(RuntimeError):
    """Something else owns what was asked for; ``owner`` is its claim."""

    def __init__(self, message: str, owner: dict[str, Any]) -> None:
        super().__init__(message)
        self.owner = owner


def alive(pid: int) -> bool:
    """Whether a process with this pid exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def normal(kind: str, key: str) -> str:
    """The canonical spelling of ``key`` for ``kind``; raises ValueError for one that is not valid."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if kind == "port":
        if not key.isdigit() or not 0 < int(key) < 65536:
            raise ValueError("a port is a number from 1 to 65535")
        return str(int(key))
    if kind in PATH_KINDS:
        expanded = Path(key).expanduser()
        if not expanded.is_absolute():
            raise ValueError("a path claim needs an absolute path")
        return os.path.realpath(expanded)
    if not WORD.match(key) or ".." in key:
        raise ValueError(f"{key!r} is not a usable {kind} name")
    return key


def _nested(a: str, b: str) -> bool:
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def _covers(claim: dict[str, Any], kind: str, key: str) -> bool:
    if kind in PATH_KINDS:
        return claim["kind"] in PATH_KINDS and _nested(claim["key"], key)
    return claim["kind"] == kind and claim["key"] == key


class Claims:
    """Claims with a TTL renewed by heartbeat; one whose pid has died is released on next look."""

    def __init__(self, base: Path, ttl_s: float, clock: Callable[[], float] = time.time,
                 on_swept: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.path = base / "claims.json"
        self.lock = base / "claims.lock"
        self.ttl_s, self.clock, self.on_swept = ttl_s, clock, on_swept

    def _load(self) -> dict[str, dict[str, Any]]:
        data = read_json(self.path, {})
        found = data.get("claims") if isinstance(data, dict) else None
        return dict(found) if isinstance(found, dict) else {}

    def _save(self, claims: dict[str, dict[str, Any]]) -> None:
        write_json(self.path, {"version": VERSION, "claims": claims})
        self.path.chmod(0o600)

    def _sweep(self, claims: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        now = self.clock()
        dead = [c for c in claims.values()
                if c["expires"] <= now or (c["pid"] and not alive(int(c["pid"])))]
        for claim in dead:
            claims.pop(f"{claim['kind']}:{claim['key']}", None)
            if self.on_swept:
                self.on_swept(claim)
        return dead

    def claim(self, who: Identity, kind: str, key: str,
              fields: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Take ``key``; ``fields`` may hold ``ttl_s``, ``pid`` and ``note``. Returns the claim
        and any dead claims that were released on the way."""
        ttl_s, pid = float((fields or {}).get("ttl_s", 0.0)), int((fields or {}).get("pid", 0))
        note = str((fields or {}).get("note", ""))
        key = normal(kind, key)
        with held(self.lock):
            claims = self._load()
            swept = self._sweep(claims)
            if swept:
                self._save(claims)
            for other in claims.values():
                if _covers(other, kind, key) and other["owner"] != who.id:
                    raise Conflict(f"{kind} {key} is held by {other['owner']} until "
                                   f"{time.strftime('%H:%M:%S', time.localtime(other['expires']))}",
                                   other)
            now = self.clock()
            made = {"kind": kind, "key": key, "owner": who.id, "pid": int(pid), "since": now,
                    "expires": now + (ttl_s or self.ttl_s), "note": note[:200]}
            claims[f"{kind}:{key}"] = made
            self._save(claims)
            return made, swept

    def release(self, who: Identity, kind: str, key: str) -> dict[str, Any]:
        """Give up a claim. Its owner, a lead or a human may."""
        key = normal(kind, key)
        with held(self.lock):
            claims = self._load()
            self._sweep(claims)
            found = claims.get(f"{kind}:{key}")
            if found is None:
                raise ValueError(f"{kind} {key} is not claimed")
            if found["owner"] != who.id and who.role == AGENT:
                raise Denied(f"{kind} {key} belongs to {found['owner']}")
            claims.pop(f"{kind}:{key}")
            self._save(claims)
            return found

    def heartbeat(self, who: Identity, ttl_s: float = 0.0) -> int:
        """Renew every claim ``who`` holds; returns how many."""
        with held(self.lock):
            claims = self._load()
            self._sweep(claims)
            mine = [c for c in claims.values() if c["owner"] == who.id]
            for claim in mine:
                claim["expires"] = self.clock() + (ttl_s or self.ttl_s)
            self._save(claims)
            return len(mine)

    def listing(self, owner: str = "", kind: str = "") -> list[dict[str, Any]]:
        """Live claims, optionally of one owner or kind."""
        with held(self.lock):
            claims = self._load()
            if self._sweep(claims):
                self._save(claims)
        return sorted((c for c in claims.values() if (not owner or c["owner"] == owner)
                       and (not kind or c["kind"] == kind)), key=lambda c: (c["kind"], c["key"]))

    def who(self, kind: str, key: str) -> dict[str, Any] | None:
        """The claim that covers ``key``, or None when nobody owns it."""
        key = normal(kind, key)
        return next((c for c in self.listing() if _covers(c, kind, key)), None)
