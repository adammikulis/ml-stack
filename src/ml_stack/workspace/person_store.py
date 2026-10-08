"""Where the person record is kept and how it is read back: a sealed, hash-chained log of what the harness hook saw the person say."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.lock import Busy, only_one
from ml_stack.sentinel.events import EventLog

__all__ = ["ATTESTATION", "DEFAULT_MINUTES", "MAX_MINUTES", "VERSION", "Authorization", "Unreadable",
           "authorizations", "bound_sessions", "consume_lock", "directory", "open_log", "records"]

ATTESTATION = "person-attestation"
VERSION = 1
DEFAULT_MINUTES = 15
MAX_MINUTES = 240
CLOCK_SKEW_S = 60.0


class Unreadable(RuntimeError):
    """The person record cannot be read or fails verification."""


def directory() -> Path:
    """The folder holding the person record."""
    return home.state("person")


def open_log(root: Path | None = None) -> EventLog:
    """The sealed log of statements, authorizations and their transitions."""
    return EventLog((root or directory()) / "statements.log", max_bytes=64_000_000, keep=4)


def records(log: EventLog | None = None) -> list[dict[str, Any]]:
    """Every record in order; raises `Unreadable` when the chain or its seal fails."""
    log = log or open_log()
    verdict = log.verify()
    if not verdict.ok:
        raise Unreadable("; ".join(verdict.problems) or "the person record failed verification")
    return log.raw()


@dataclass(frozen=True, slots=True)
class Authorization:
    """One authorization with its state worked out from the transitions that follow it."""

    id: str
    kind: str
    target: str
    session_id: str
    project: str
    statement_seq: int
    how: str
    uses: int
    used: int
    expires: float
    state: str
    ts: float

    def live(self, now: float) -> bool:
        """Whether it can still be consumed at ``now``."""
        return self.state == "live" and self.used < self.uses and now < self.expires


def authorizations(rows: list[dict[str, Any]], now: float | None = None) -> list[Authorization]:
    """The authorizations in ``rows`` with their current state."""
    now = time.time() if now is None else now
    held: dict[str, dict[str, Any]] = {}
    used: dict[str, int] = {}
    ended: dict[str, str] = {}
    for row in rows:
        what = row.get("type")
        if what == "authorization":
            held[row["id"]] = row
        elif what == "transition" and row.get("auth_id") in held:
            if row.get("state") == "used":
                used[row["auth_id"]] = used.get(row["auth_id"], 0) + 1
            else:
                ended.setdefault(row["auth_id"], str(row["state"]))
    out = []
    for key, row in held.items():
        state = ended.get(key) or ("used" if used.get(key, 0) >= row["uses"] else "live")
        if state == "live" and now >= row["expires"]:
            state = "expired"
        out.append(Authorization(key, row["kind"], row["target"], row["session_id"], row["project"],
                                 row["statement_seq"], row["how"], row["uses"], used.get(key, 0),
                                 row["expires"], state, row["ts"]))
    return out


def newest_clock(rows: list[dict[str, Any]], clock: Callable[[], float] = time.time) -> bool:
    """Whether no record is dated after the present."""
    return all(r.get("ts", 0) <= clock() + CLOCK_SKEW_S for r in rows)


@contextmanager
def consume_lock(log: EventLog) -> Iterator[None]:
    """Hold the lock that orders consuming, revoking and expiring authorizations."""
    try:
        with only_one(log.path.with_name(log.path.name + ".consume"), timeout=5.0, announce=lambda _: None):
            yield
    except Busy as error:
        raise Unreadable(str(error)) from error


def bound_sessions(rows: list[dict[str, Any]]) -> dict[tuple[int, float], str]:
    """The session each recorded harness process `(pid, create_time)` was last bound to."""
    return {(r["pid"], r["created"]): r["session_id"] for r in rows if r.get("type") == "binding"}
