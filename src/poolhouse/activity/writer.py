"""`record`: the one call every feed makes. It never raises into the caller; a record that
cannot be written is counted."""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Unpack

import psutil

from poolhouse import home, keystore
from poolhouse.activity.log import ActivityLog, Limits
from poolhouse.activity.schema import Said, build
from poolhouse.files import read_json, write_json
from poolhouse.sentinel.human import AGENT_MARKERS, protect

__all__ = ["ENV_OFF", "ENV_RETENTION", "FAILURES", "bind_session", "directory", "drops", "log",
           "record", "session"]

logger = logging.getLogger("poolhouse.activity")
logger.addHandler(logging.NullHandler())

ENV_OFF = "POOLHOUSE_ACTIVITY"
ENV_RETENTION = "POOLHOUSE_ACTIVITY_RETENTION_DAYS"
ENV_ACTOR = "POOLHOUSE_ACTOR"
ENV_SESSION = "POOLHOUSE_SESSION"
FAILURES = (OSError, ValueError, RuntimeError, TypeError, KeyError, AttributeError,
            keystore.KeystoreError)
"""What `record` swallows: a full disk, a locked key or a damaged file never breaks a caller."""

_LOCAL = threading.local()
_LOCK = threading.Lock()
_LOGS: dict[str, ActivityLog] = {}
_STATE: dict[str, Any] = {"session": "", "dropped": 0, "owed": 0, "told": set()}


def directory() -> Path:
    """This user's activity directory under the state root."""
    return home.state("activity", "u-" + keystore.os_user().partition(":")[0])


def log() -> ActivityLog:
    """The log of this user at the state root in force now."""
    where = directory()
    with _LOCK:
        if str(where) not in _LOGS:
            protect(where)
            days = _days()
            _LOGS[str(where)] = ActivityLog(where, limits=Limits(retention_s=days * 86400.0))
        return _LOGS[str(where)]


def _days() -> float:
    try:
        return max(0.0, float(os.environ.get(ENV_RETENTION, "90")))
    except ValueError:
        return 90.0


def session() -> str:
    """The session id records carry: bound by the chat or run, else one per process."""
    with _LOCK:
        if not _STATE["session"]:
            _STATE["session"] = os.environ.get(ENV_SESSION) or "p-" + secrets.token_hex(4)
        return str(_STATE["session"])


def bind_session(name: str) -> None:
    """Make ``name`` the session id of every record this process writes from now on."""
    with _LOCK:
        _STATE["session"] = name


def _actor() -> str:
    named = os.environ.get(ENV_ACTOR)
    if named:
        return named
    marked = next((m for m in AGENT_MARKERS if os.environ.get(m)), "")
    return "agent" if marked else "system"


def _mode() -> str:
    if os.environ.get(ENV_OFF, "").strip().lower() != "off":
        return "on"
    return "refused" if any(os.environ.get(m) for m in AGENT_MARKERS) else "off"


def drops() -> dict[str, Any]:
    """Records dropped by this user's processes: the total, when last, and why."""
    held = read_json(directory() / "drops.json", {})
    return held if isinstance(held, dict) else {}


def _dropped(cause: str, now: float) -> None:
    with _LOCK:
        _STATE["dropped"] += 1
        _STATE["owed"] += 1
    logger.debug("activity record dropped: %s", cause)
    try:
        held = drops()
        write_json(directory() / "drops.json", {
            "dropped": int(held.get("dropped", 0)) + 1, "last": round(now, 3), "cause": cause,
            "writer_pid": os.getpid(), "writer_started": psutil.Process(os.getpid()).create_time(),
            "writer_at": time.time()})
    except (*FAILURES, psutil.Error):
        return


def _put(kind: str, *, ts: float, actor: str, session: str, **said: Unpack[Said]) -> None:
    """Write one record, preceded by a gap record when earlier ones were dropped."""
    target = log()
    with _LOCK:
        owed = int(_STATE["owed"])
    if owed:
        target.add(build("activity.gap", ts=ts, actor="system", session=session,
                         meta={"dropped": owed, "cause": str(drops().get("cause", ""))}))
        with _LOCK:
            _STATE["owed"] -= owed
    target.add(build(kind, ts=ts, actor=actor, session=session, **said))


def _announce(mode: str, now: float) -> bool:
    """Write the one record that says the log was switched off (or that an agent tried);
    true when records are to be written after it."""
    with _LOCK:
        first = mode not in _STATE["told"]
        _STATE["told"].add(mode)
    if first:
        kind = "activity.off" if mode == "off" else "activity.off_refused"
        try:
            _put(kind, ts=now, actor=_actor(), session=session(), outcome=mode)
        except FAILURES as exc:
            _dropped(type(exc).__name__, now)
    return mode == "refused"


def record(kind: str, *, actor: str | None = None, ts: float | None = None,
           **said: Unpack[Said]) -> bool:
    """Append one activity record (``subject``, ``outcome``, ``refs``, ``meta``); false when it
    was not written. Never raises."""
    if getattr(_LOCAL, "busy", False):
        return False
    _LOCAL.busy = True
    now = time.time() if ts is None else ts
    try:
        mode = _mode()
        if mode != "on" and not _announce(mode, now):
            return False
        _put(kind, ts=now, actor=actor or _actor(), session=session(), **said)
        return True
    except FAILURES as exc:
        _dropped(type(exc).__name__, now)
        return False
    finally:
        _LOCAL.busy = False
