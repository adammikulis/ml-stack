"""A heads-up on the person's own screen when something is quarantined.

The notification has two real buttons: ``Review…`` opens a terminal running
``ml-stack-security review``, and ``Dismiss`` does nothing. Nothing here can release, purge or
answer a prompt: this module never touches a grant, a store method that changes a record, or
the review screen's keys, and the only text it hands the desktop is the safe, bounded
sentence from `explain`. At most one notice per subject per hour, and one burst notice when
many are held at once. ``ML_STACK_SENTINEL_NOTIFY=off`` turns it off but needs
``ML_STACK_SENTINEL_NOTIFY_BECAUSE``; the off is logged as an event.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from ml_stack import desktop
from ml_stack.files import write_text
from ml_stack.lock import only_one
from ml_stack.sentinel import explain
from ml_stack.sentinel.events import Bus, Event, Severity
from ml_stack.sentinel.launcher import open_review
from ml_stack.sentinel.store import Record

__all__ = ["BECAUSE", "BUTTONS", "ENV", "HOUR_S", "HeadsUp", "Wires"]

ENV = "ML_STACK_SENTINEL_NOTIFY"
BECAUSE = "ML_STACK_SENTINEL_NOTIFY_BECAUSE"
REVIEW, DISMISS = "Review…", "Dismiss"
BUTTONS = (DISMISS, REVIEW)
"""The two buttons; the safe one first, so Escape and closing the window dismiss."""

HOUR_S = 3600.0
BURST_AFTER = 3
"""Individual notices allowed in a minute before the rest become one burst notice."""

logger = logging.getLogger("ml_stack.sentinel")
Choose = Callable[[str, str, tuple[str, str]], str]


def _in_thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name="sentinel-heads-up", daemon=True).start()


class Wires(NamedTuple):
    """What a `HeadsUp` calls out to: ``choose`` shows the buttons and returns the label
    pressed (None: the desktop's), ``opener`` opens the review window, ``spawn`` runs the
    blocking dialog off the caller's thread."""

    choose: Choose | None = None
    opener: Callable[[], bool] = open_review
    spawn: Callable[[Callable[[], None]], None] = _in_thread


class HeadsUp:
    """Raises the notification for a quarantine."""

    def __init__(self, state: Path, *, clock: Callable[[], float], bus: Bus | None = None,
                 env: Mapping[str, str] | None = None, wires: Wires | None = None) -> None:
        self.state, self.clock, self.bus = Path(state), clock, bus
        self.env = os.environ if env is None else env
        self.wires = wires or Wires()
        self.choose, self.opener, self.spawn = self.wires
        self._reported_off = False

    # -- the hook ------------------------------------------------------------------
    def on_quarantine(self, record: Record) -> None:
        """Store hook: tell the person unless it is off, rate-limited, or the person's own
        doing. Never raises."""
        try:
            if record.history and record.history[-1].get("actor") == "human":
                return
            if self._off() or (self.choose is None and desktop.which_way() == "none"):
                return
            title, body = self._compose(record)
            if title:
                self._raise(title, body)
        except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
            logger.warning("sentinel heads-up failed: %s", type(exc).__name__)

    def _off(self) -> bool:
        if self.env.get(ENV, "").strip().lower() != "off":
            return False
        because = self.env.get(BECAUSE, "").strip()
        if not because:
            return False
        if not self._reported_off and self.bus is not None:
            self._reported_off = True
            self.bus.emit(Event("sentinel.notify_off", Severity.WARNING, "heads_up", "",
                                {"because": because}, self.clock()))
        return True

    # -- what to say, and whether it is time to -------------------------------------
    def _compose(self, record: Record) -> tuple[str, str]:
        now = self.clock()
        item = explain.describe(record, 1, now)
        with only_one(self.state.with_name(self.state.name + ".lock")):
            memo = self._load(now)
            subject = f"{record.kind}:{record.key}"[:200]
            if subject in memo["subjects"]:
                return "", ""
            memo["subjects"][subject] = now
            memo["recent"].append(now)
            burst = len([t for t in memo["recent"] if now - t < 60]) > BURST_AFTER
            if burst and now - memo["burst"] < HOUR_S:
                self._save(memo)
                return "", ""
            if burst:
                memo["burst"] = now
            self._save(memo)
        if burst:
            return ("Several things were just quarantined",
                    "ml-stack-security review lists them. Nothing has been released.")
        return (f"ml-stack quarantined {item.name}", f"{item.why} {item.blocks}")

    def _load(self, now: float) -> dict[str, Any]:
        try:
            data = json.loads(self.state.read_text())
        except (OSError, ValueError):
            data = {}
        subjects = {k: t for k, t in dict(data.get("subjects", {})).items()
                    if isinstance(t, int | float) and now - t < HOUR_S}
        recent = [t for t in data.get("recent", []) if isinstance(t, int | float) and now - t < 60]
        burst = data.get("burst", 0.0)
        return {"version": 1, "subjects": dict(list(subjects.items())[-200:]), "recent": recent,
                "burst": burst if isinstance(burst, int | float) else 0.0}

    def _save(self, memo: dict[str, Any]) -> None:
        write_text(self.state, json.dumps(memo, sort_keys=True))

    # -- the buttons ---------------------------------------------------------------
    def _raise(self, title: str, body: str) -> None:
        def ask() -> None:
            try:
                answer = (self.choose or _desktop)(title, body, BUTTONS)
                if answer == REVIEW:
                    self.opener()
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("sentinel heads-up failed: %s", type(exc).__name__)

        self.spawn(ask)


def _desktop(title: str, body: str, buttons: tuple[str, str]) -> str:
    return desktop.choose(title, body, buttons)
