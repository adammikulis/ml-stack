"""Consuming an authorization the person spoke: a guard asks for a kind and the target it derived itself."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ml_stack.lock import Busy, only_one
from ml_stack.sentinel.events import EventLog
from ml_stack.workspace import person_record, person_store

__all__ = ["Identity", "NotAuthorized", "consume", "live"]


class NotAuthorized(RuntimeError):
    """No live authorization matches, or the record cannot be trusted."""


@dataclass(frozen=True, slots=True)
class Identity:
    """The session a guard acts in and, for a subagent, its label; a label never widens what the parent holds."""

    session_id: str
    label: str = ""


def live(kind: str, target: str, identity: Identity, *, log: EventLog | None = None,
         now: float | None = None) -> list[person_store.Authorization]:
    """The authorizations ``identity`` could consume for ``kind`` on ``target`` now."""
    now = time.time() if now is None else now
    rows = person_store.records(log)
    if not person_store.newest_clock(rows, lambda: now):
        raise person_store.Unreadable("the person record has entries dated in the future")
    return [a for a in person_store.authorizations(rows, now)
            if a.live(now) and a.kind == kind and a.target == target
            and a.session_id == identity.session_id and identity.session_id]


def consume(kind: str, target: str, identity: Identity, *, log: EventLog | None = None,
            now: float | None = None) -> str:
    """Use one authorization of ``kind`` for exactly ``target`` and return its id; raise `NotAuthorized`."""
    log = log or person_store.open_log()
    try:
        with only_one(log.path.with_name(log.path.name + ".consume"), timeout=5.0, announce=lambda _: None):
            found = live(kind, target, identity, log=log, now=now)
            if not found:
                raise NotAuthorized(f"no live {kind} authorization for {target} in this session")
            chosen = min(found, key=lambda a: a.ts)
            person_record.mark_used(log, chosen.id, by=identity.label or "main", target=target)
            return chosen.id
    except (person_store.Unreadable, Busy, OSError, ValueError) as error:
        raise NotAuthorized(f"the person record is unavailable: {error}") from error
