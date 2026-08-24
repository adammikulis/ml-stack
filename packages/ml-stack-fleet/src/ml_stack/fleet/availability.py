"""When a machine will take work, and when it is somebody's desk.

A box in a cupboard can train at any hour. A box someone works on cannot, and the failure
without this is specific and annoying: a training run started overnight is still holding
the GPU at 9am, so the machine someone needs is slow all morning and the only fix is to
notice and kill it.

Two mechanisms, deliberately separate:

**Windows** are what a machine says about itself -- "not between nine and five on
weekdays". They are recurring, local wall-clock, and configured on the box, because the
box is the thing that knows whose desk it is on.

**Reservations** are what one machine asks of another -- "hold this box for me until
half past". They are one-off and come from the network.

**A pause** is the person at the keyboard saying "not now" -- they have started a game,
or a render, or they just want their machine back. It beats both of the above, takes
effect immediately, and unlike a window it is not something anyone has to have predicted.
It survives a restart, because a pause that quietly lifted when the box rebooted would
hand the GPU back at the worst possible moment.

Both answer the same question, ``open_at``, and the daemon asks it before it starts a
job rather than before it queues one. Queued work waits for the window to open; it is not
refused, because "come back at six" is a thing a scheduler can act on and a rejection is
not.

Local time on purpose. "The working day" is a fact about where the machine is, and a box
in another timezone has a different one -- converting them to a shared clock would make
every window mean the wrong thing on half the fleet.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = ["Availability", "Reservation", "Window", "parse_window"]

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_SPEC = re.compile(
    r"^\s*(?P<days>[a-z][a-z,\-\s]*?)?\s*(?P<from>\d{1,2}:\d{2})\s*-\s*(?P<to>\d{1,2}:\d{2})"
    r"\s*(?:#\s*(?P<note>.*?))?\s*$", re.I)


def _minutes(hhmm: str) -> int:
    hours, _, mins = hhmm.partition(":")
    total = int(hours) * 60 + int(mins)
    if not 0 <= total <= 24 * 60:
        raise ValueError(f"{hhmm!r} is not a time of day")
    return total


def _day_set(spec: str | None) -> tuple[int, ...]:
    if not spec or spec.lower() in ("daily", "all", "everyday"):
        return tuple(range(7))
    out: set[int] = set()
    for part in spec.lower().replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                start, end = DAYS.index(a.strip()[:3]), DAYS.index(b.strip()[:3])
            except ValueError:
                raise ValueError(f"{part!r} is not a day range like 'mon-fri'") from None
            # Wraps: "fri-mon" is Friday through Monday, not an empty range.
            i = start
            while True:
                out.add(i)
                if i == end:
                    break
                i = (i + 1) % 7
        else:
            try:
                out.add(DAYS.index(part[:3]))
            except ValueError:
                raise ValueError(f"{part!r} is not a day like 'mon'") from None
    return tuple(sorted(out))


@dataclass(frozen=True, slots=True)
class Window:
    """A recurring span of local wall-clock time, and what it means."""

    days: tuple[int, ...]
    start_min: int
    end_min: int
    busy: bool = True
    """True means "do not start work here". The common case is blocking out a working
    day, so it is the default -- a window someone writes down is nearly always one they
    want protected."""
    note: str = ""

    def covers(self, when: datetime) -> bool:
        """Whether this window is in force at ``when``.

        A window whose end is not after its start wraps past midnight -- "22:00-06:00"
        is one span through the night, not an empty one. Getting this wrong silently
        disables the overnight window, which is the one people most want.
        """
        minute = when.hour * 60 + when.minute
        if self.end_min > self.start_min:
            return when.weekday() in self.days and self.start_min <= minute < self.end_min
        # Wrapping: the tail belongs to the day the window STARTED on.
        if minute >= self.start_min:
            return when.weekday() in self.days
        return (when.weekday() - 1) % 7 in self.days and minute < self.end_min

    def spec(self) -> str:
        """The canonical text form, which ``parse_window`` reads back exactly.

        Separate from ``describe`` because the two have different jobs and conflating
        them cost a real bug: a saved schedule written in the prose form did not parse
        on the next start, so a machine silently lost the working hours it was supposed
        to be protecting.
        """
        days = "daily" if len(self.days) == 7 else ",".join(DAYS[d] for d in self.days)
        out = (f"{days} {self.start_min // 60:02d}:{self.start_min % 60:02d}"
               f"-{self.end_min // 60:02d}:{self.end_min % 60:02d}")
        return f"{out} # {self.note}" if self.note else out

    def describe(self) -> str:
        """For a person to read. Never fed back to the parser -- see ``spec``."""
        days = ("every day" if len(self.days) == 7
                else " ".join(DAYS[d] for d in self.days))
        return (f"{days} {self.start_min // 60:02d}:{self.start_min % 60:02d}"
                f"-{self.end_min // 60:02d}:{self.end_min % 60:02d}"
                f"{' (' + self.note + ')' if self.note else ''}")


def parse_window(spec: str, *, busy: bool = True, note: str = "") -> Window:
    """``"mon-fri 09:00-17:00"`` into a Window. ``"22:00-06:00"`` wraps midnight."""
    match = _SPEC.match(spec)
    if not match:
        raise ValueError(
            f"{spec!r} should look like 'mon-fri 09:00-17:00' or '22:00-06:00'")
    return Window(days=_day_set(match.group("days")),
                  start_min=_minutes(match.group("from")),
                  end_min=_minutes(match.group("to")),
                  busy=busy, note=note or (match.group("note") or "").strip())


@dataclass(frozen=True, slots=True)
class Reservation:
    """One machine holding another for a while."""

    holder: str
    until: float
    reason: str = ""

    @property
    def live(self) -> bool:
        return self.until > time.time()

    def public(self) -> dict[str, Any]:
        return {"holder": self.holder, "until": self.until, "reason": self.reason,
                "seconds_left": max(0.0, round(self.until - time.time(), 1))}


@dataclass
class Availability:
    """Whether this box will start work right now, and if not, when it will."""

    windows: list[Window] = field(default_factory=list)
    reservation: Reservation | None = None
    max_hold_s: float = 12 * 3600
    """Longest a peer may hold this box. A reservation with no ceiling is a way to take
    a machine out of the fleet permanently by accident."""
    paused_until: float | None = None
    """``None`` when running. A timestamp when paused for a while, and ``math.inf`` when
    paused until someone says otherwise. Indefinite is the honest default for "I am
    using this now": nobody knows how long a game lasts, and a pause that expired
    mid-session would be worse than one that has to be lifted by hand."""
    paused_reason: str = ""

    # -- windows ---------------------------------------------------------
    def open_at(self, when: datetime | None = None) -> bool:
        """Whether a *new* job may start now, ignoring reservations."""
        when = when or datetime.now()
        allowed = None
        for window in self.windows:
            if window.covers(when):
                # An explicit allow beats a block, so "never on weekdays, except at
                # lunchtime" is expressible without inventing precedence rules.
                if not window.busy:
                    return True
                allowed = False
        return True if allowed is None else allowed

    def blocking(self, when: datetime | None = None) -> Window | None:
        when = when or datetime.now()
        if self.open_at(when):
            return None
        return next((w for w in self.windows if w.busy and w.covers(when)), None)

    def opens_at(self, when: datetime | None = None, *, horizon_h: int = 24 * 8
                 ) -> datetime | None:
        """When this box next takes work, or None if nothing in the horizon says so.

        Stepped a minute at a time rather than solved algebraically: windows may overlap,
        allow may override busy, and the wrap-past-midnight case makes a closed-form
        answer the kind of code that is wrong for one hour a week and nobody notices.
        """
        when = when or datetime.now()
        if self.open_at(when):
            return when
        probe = when.replace(second=0, microsecond=0)
        for _ in range(horizon_h * 60):
            probe += timedelta(minutes=1)
            if self.open_at(probe):
                return probe
        return None

    # -- reservations ----------------------------------------------------
    def reserve(self, holder: str, seconds: float, reason: str = "") -> Reservation:
        held = self.held_by()
        if held and held.holder != holder:
            raise PermissionError(
                f"held by {held.holder} for another "
                f"{held.public()['seconds_left']:.0f}s")
        seconds = max(1.0, min(float(seconds), self.max_hold_s))
        self.reservation = Reservation(holder, time.time() + seconds, reason)
        return self.reservation

    def release(self, holder: str) -> bool:
        held = self.held_by()
        if held is None or held.holder != holder:
            return False
        self.reservation = None
        return True

    def held_by(self) -> Reservation | None:
        if self.reservation is not None and self.reservation.live:
            return self.reservation
        self.reservation = None
        return None

    # -- the manual toggle ----------------------------------------------
    def pause(self, *, minutes: float | None = None, reason: str = "") -> None:
        """Stop taking work now. ``minutes=None`` means until someone resumes."""
        self.paused_until = (math.inf if minutes is None
                             else time.time() + max(1.0, minutes) * 60)
        self.paused_reason = reason

    def resume(self) -> None:
        self.paused_until = None
        self.paused_reason = ""

    @property
    def paused(self) -> bool:
        if self.paused_until is None:
            return False
        if self.paused_until <= time.time():
            self.paused_until = None
            self.paused_reason = ""
            return False
        return True

    def may_start(self, who: str = "", when: datetime | None = None) -> tuple[bool, str]:
        """The single question the job runner asks. Returns ``(allowed, why not)``.

        The pause is checked first because it is the only one of the three that means
        someone is at the keyboard right now.
        """
        if self.paused:
            if self.paused_until == math.inf:
                until = "until it is switched back on"
            else:
                mins = max(1, round((self.paused_until - time.time()) / 60))
                until = f"for about {mins} more minute{'s' if mins != 1 else ''}"
            why = self.paused_reason or "someone is using this machine"
            return False, f"paused: {why}, {until}"
        held = self.held_by()
        if held is not None and who != held.holder:
            return False, (f"reserved by {held.holder} for another "
                           f"{held.public()['seconds_left']:.0f}s")
        blocked = self.blocking(when)
        if blocked is not None:
            nxt = self.opens_at(when)
            when_txt = nxt.strftime("%a %H:%M") if nxt else "an unknown time"
            return False, (f"this machine is in use ({blocked.describe()}); "
                           f"work resumes {when_txt}")
        return True, ""

    # -- reporting -------------------------------------------------------
    def public(self, when: datetime | None = None) -> dict[str, Any]:
        allowed, why = self.may_start(when=when)
        # A pause has no scheduled end, so do not invent one from the windows: showing
        # "resumes at 17:00" for a machine someone paused indefinitely is a promise the
        # box has no way to keep.
        nxt = None if allowed or self.paused else self.opens_at(when)
        return {"available": allowed,
                "unavailable_because": why,
                "paused": self.paused,
                "paused_until": (None if not self.paused
                                 or self.paused_until == math.inf
                                 else self.paused_until),
                "next_open": nxt.timestamp() if nxt else None,
                "windows": [w.describe() for w in self.windows],
                "reserved": (self.held_by().public() if self.held_by() else None)}

    # -- config ----------------------------------------------------------
    @classmethod
    def from_specs(cls, busy: list[str] = (), free: list[str] = ()) -> "Availability":
        return cls(windows=[parse_window(s) for s in busy]
                   + [parse_window(s, busy=False) for s in free])

    @classmethod
    def load(cls, path: Path | str) -> "Availability":
        """Read windows from a JSON file, so the UI can edit what the CLI set."""
        p = Path(path).expanduser()
        if not p.exists():
            return cls()
        try:
            raw = json.loads(p.read_text())
        except (OSError, ValueError):
            # An unreadable schedule must not take the box out of the fleet. Available
            # is the safe failure: someone notices a run at the wrong time far sooner
            # than they notice a machine that quietly never accepts work.
            return cls()
        out = cls.from_specs(raw.get("busy", []), raw.get("free", []))
        held = raw.get("paused_until")
        if held is not None:
            out.paused_until = math.inf if held == "forever" else float(held)
            out.paused_reason = raw.get("paused_reason", "")
        return out

    def save(self, path: Path | str) -> Path:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, Any] = {
            "busy": [w.spec() for w in self.windows if w.busy],
            "free": [w.spec() for w in self.windows if not w.busy]}
        if self.paused:
            body["paused_until"] = ("forever" if self.paused_until == math.inf
                                    else self.paused_until)
            body["paused_reason"] = self.paused_reason
        p.write_text(json.dumps(body, indent=2))
        return p
