"""Sources, events and the two-timescale score: pure functions over one source's record."""

from __future__ import annotations

import ipaddress
import re
import urllib.parse
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "BAD_AT",
    "BAD_FLOOR",
    "CLEAN_GAP_S",
    "ESTABLISHED_AGE_S",
    "ESTABLISHED_CLEAN",
    "EVENTS",
    "KINDS",
    "LONG_HALF_LIFE_S",
    "RECOVER_CLEAN",
    "SHORT_WINDOW_S",
    "TRAIT_EVENTS",
    "WATCH_AT",
    "WATCH_FLOOR",
    "Rec",
    "State",
    "clean_run",
    "decay",
    "misbehaved",
    "normal",
    "observe_trait",
    "short_score",
]

KINDS = ("ip", "host", "url", "peer", "repo", "hash", "connector")
EVENTS = {"denial": 1.0, "scan_hit": 3.0, "injection_flagged": 1.0, "cert_or_key_change": 3.0,
          "hash_change": 3.0, "redirect_change": 0.5, "ip_change": 0.5}
"""Bad events and the weight each adds to a source's score."""
TRAIT_EVENTS = {"cert": "cert_or_key_change", "hash": "hash_change", "redirect": "redirect_change",
                "ip_range": "ip_change"}
"""Stable traits whose change is an event; any other trait is recorded and never scored."""

WATCH_AT, BAD_AT = 1.0, 3.0
"""Short-term score at which a source is watched, and at which a source with no standing is bad."""
SHORT_WINDOW_S = 6 * 3600.0
LONG_HALF_LIFE_S = 30 * 86400.0
WATCH_FLOOR, BAD_FLOOR = 0.5, 1.0
"""The long-term score of a watched or bad source never decays below these."""
ESTABLISHED_CLEAN, ESTABLISHED_AGE_S = 10, 3 * 86400.0
CLEAN_GAP_S = 300.0
"""Clean runs closer together than this count once."""
RECOVER_CLEAN = 10
"""Clean runs in a row that return a watched source to where its history puts it; a bad
source needs three times as many to reach watch."""
DEDUP_S = 60.0
MAX_RECENT, MAX_TRAIT_VALUES, CLEAN_CAP = 24, 8, 100_000


class State(StrEnum):
    UNKNOWN = "unknown"
    ESTABLISHED = "established"
    WATCH = "watch"
    BAD = "bad"


_HOST = re.compile(r"[a-z0-9_]([a-z0-9_.-]{0,251}[a-z0-9_])?")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,63}")
_REPO = re.compile(r"[a-z0-9][a-z0-9_.-]{0,60}(/[a-z0-9_.-]{1,80}){0,2}")
_HEX = re.compile(r"[0-9a-f]{64}")


def normal(kind: str, key: str) -> str:
    """``key`` in the one spelling ``kind`` has; ``ValueError`` when it is not a valid one."""
    text = str(key).strip()
    if kind == "host":
        name = text.lower().rstrip(".").encode("idna").decode("ascii") if text else ""
        if _HOST.fullmatch(name):
            return name
    elif kind == "ip":
        return str(ipaddress.ip_address(text.split("%")[0].strip("[]")))
    elif kind == "url":
        parts = urllib.parse.urlsplit(text)
        host = (parts.hostname or "").lower()
        if parts.scheme in ("http", "https") and _HOST.fullmatch(host):
            return f"{parts.scheme}://{host}{parts.path[:200] or '/'}"
    elif kind == "repo" and _REPO.fullmatch(text.lower()) and ".." not in text:
        return text.lower()
    elif kind == "hash":
        digest = text.lower().removeprefix("sha256:")
        if _HEX.fullmatch(digest):
            return digest
    elif kind in ("peer", "connector") and _NAME.fullmatch(text):
        return text
    raise ValueError(f"not a {kind} name")


@dataclass(slots=True)
class Rec:
    """What is held about one source."""

    kind: str
    key: str
    first: float
    last: float
    state: str = State.UNKNOWN
    since: float = 0.0
    clean: int = 0
    streak: int = 0
    last_clean: float = 0.0
    long_bad: float = 0.0
    long_at: float = 0.0
    recent: list[list[Any]] = field(default_factory=list)
    traits: dict[str, list[str]] = field(default_factory=dict)
    notice: str = ""
    events: list[str] = field(default_factory=list)

    @classmethod
    def new(cls, kind: str, key: str, now: float) -> Rec:
        return cls(kind, key, now, now, since=now, long_at=now)

    @classmethod
    def from_attrs(cls, attrs: dict[str, Any]) -> Rec:
        return cls(**{name: attrs[name] for name in cls.__dataclass_fields__ if name in attrs})

    def attrs(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def floor_of(state: str) -> float:
    return {State.BAD: BAD_FLOOR, State.WATCH: WATCH_FLOOR}.get(State(state), 0.0)


def decay(value: float, dt: float, floor: float = 0.0) -> float:
    """``value`` after ``dt`` seconds of the long-term half-life, never lower than ``floor``."""
    if value <= floor:
        return value
    return max(floor, value * 0.5 ** (max(dt, 0.0) / LONG_HALF_LIFE_S))


def long_score(rec: Rec, now: float) -> float:
    """The long-term badness: slow, weeks long, floored for a watched or bad source."""
    return decay(rec.long_bad, now - rec.long_at, floor_of(rec.state))


def short_score(rec: Rec, now: float) -> float:
    """The badness of the events in the last few hours."""
    return sum(float(w) for ts, _, w in rec.recent if now - float(ts) <= SHORT_WINDOW_S)


def _move(rec: Rec, state: State, now: float) -> None:
    if rec.state != state:
        rec.state, rec.since = state, now


def misbehaved(rec: Rec, event: str, now: float, scale: float = 1.0) -> bool:
    """Fold one bad ``event`` into ``rec``. True when it is a divergence: a source in good
    standing that is stepped down to watch by fresh evidence. A repeat of the same event inside
    a minute is not counted again."""
    if event not in EVENTS:
        raise ValueError(f"unknown event {event!r}")
    rec.last = now
    if any(r[1] == event and now - float(r[0]) < DEDUP_S for r in rec.recent):
        return False
    weight = EVENTS[event] * max(0.0, min(scale, 3.0))
    rec.long_bad, rec.long_at = long_score(rec, now) + weight, now
    kept = [r for r in rec.recent if now - float(r[0]) <= SHORT_WINDOW_S]
    rec.recent = [*kept, [now, event, weight]][-MAX_RECENT:]
    rec.streak = 0
    short, before = short_score(rec, now), State(rec.state)
    if before == State.ESTABLISHED:
        if short >= WATCH_AT:
            _move(rec, State.WATCH, now)
            rec.notice = "queued"
            return True
    elif before in (State.UNKNOWN, State.WATCH):
        if short >= BAD_AT:
            _move(rec, State.BAD, now)
        elif short >= WATCH_AT and before == State.UNKNOWN:
            _move(rec, State.WATCH, now)
    return False


def _standing(rec: Rec, now: float) -> State:
    """Where its history puts a source with nothing recent against it."""
    old = now - rec.first >= ESTABLISHED_AGE_S
    return State.ESTABLISHED if rec.clean >= ESTABLISHED_CLEAN and old else State.UNKNOWN


def clean_run(rec: Rec, now: float, recover: int = RECOVER_CLEAN) -> None:
    """Count one clean interaction (at most one in ``CLEAN_GAP_S``) and move the state it earns."""
    rec.last = now
    if rec.last_clean and now - rec.last_clean < CLEAN_GAP_S:
        return
    rec.last_clean = now
    rec.clean, rec.streak = min(rec.clean + 1, CLEAN_CAP), rec.streak + 1
    rec.long_bad, rec.long_at = long_score(rec, now), now
    state = State(rec.state)
    if state == State.UNKNOWN and short_score(rec, now) < WATCH_AT:
        _move(rec, _standing(rec, now), now)
    elif state == State.WATCH and rec.streak >= recover:
        rec.recent, rec.notice = [], ""
        _move(rec, _standing(rec, now), now)
    elif state == State.BAD and rec.streak >= 3 * recover:
        rec.recent, rec.streak = [], 0
        _move(rec, State.WATCH, now)


def observe_trait(rec: Rec, name: str, value: str) -> str:
    """Record ``value`` among the values seen for trait ``name``. The event name when the trait
    had a baseline and ``value`` is not in it, else an empty string."""
    seen = rec.traits.setdefault(name, [])
    if value in seen:
        return ""
    seen.append(value[:128])
    del seen[:-MAX_TRAIT_VALUES]
    return TRAIT_EVENTS.get(name, "") if len(seen) > 1 else ""
