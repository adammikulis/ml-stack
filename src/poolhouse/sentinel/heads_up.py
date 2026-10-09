"""The one dialog sentinel puts on the person's screen, with the buttons that deal with it.

Only a quarantine that means something changed or was forged (`NEEDS_PERSON`) is asked
about; a watch, a missing file or a person's own hold is not. One dialog is open at a time on
the machine (a lock file beside the state, taken without waiting), a cooldown follows each,
and everything held then is in it. ``Release`` releases exactly the subjects it lists, ``Keep
held`` stops asking about them, ``Later`` asks again in a few hours. The release is the
click: made by the process that put the dialog up, for the ids it listed, when the answer is
exactly `RELEASE` and the environment has no agent marker (`human.mint_clicked`, called
nowhere else). The text is fixed sentences from `explain` and names it has made safe.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from poolhouse import desktop, requests
from poolhouse.files import write_text
from poolhouse.lock import Busy, only_one
from poolhouse.requests import dialog
from poolhouse.sentinel import explain, human
from poolhouse.sentinel.events import Bus, Event, Severity
from poolhouse.sentinel.store import Record, State, Store

__all__ = ["BUTTONS", "COOLDOWN_S", "KEEP", "LATER", "LATER_S", "NEEDS_PERSON", "RELEASE",
           "SETTLE_S", "SHOWN_MOST", "HeadsUp", "Wires", "needs_person"]

LATER, KEEP, RELEASE = "Later", "Keep held", "Release"
BUTTONS = (LATER, KEEP, RELEASE)
"""The three buttons; the safe one first, so Escape and closing the window mean Later."""

COOLDOWN_S = 600.0
LATER_S = 4 * 3600.0
SHOWN_MOST = 4
SETTLE_S = 2.0
"""Seconds a dialog waits after the first quarantine, so what arrives with it is in the same dialog."""

NEEDS_PERSON = frozenset({
    "integrity.content_changed", "integrity.binary_mismatch", "integrity.manifest_mismatch",
    "integrity.link_retargeted", "peer.forged_traffic", "honey.token_seen", "honey.path_named",
    "honey.tool_called", "honey.endpoint_hit", "guard.tainted", "guard.tainted_text",
})
"""Finding kinds whose quarantine can only be judged by a person; every other quarantine is
shown by the status line and `poolhouse-security review` and never by a dialog."""

_DECOY_REASONS = ("carries a decoy value", "model output")
logger = logging.getLogger("poolhouse.sentinel")
Choose = Callable[[str, str, tuple[str, ...]], str]


def _in_thread(work: Callable[[], object]) -> None:
    def settled() -> None:
        time.sleep(SETTLE_S)
        work()

    threading.Thread(target=settled, name="sentinel-heads-up", daemon=True).start()


class Wires(NamedTuple):
    """What a `HeadsUp` calls out to: ``choose`` shows the buttons and returns the label
    pressed (None: the desktop's), ``spawn`` runs the blocking dialog off the caller's thread,
    ``env`` is the environment read for ``POOLHOUSE_NOTIFY`` and the agent markers (None: the
    process's)."""

    choose: Choose | None = None
    spawn: Callable[[Callable[[], object]], None] = _in_thread
    env: Mapping[str, str] | None = None


def needs_person(record: Record) -> bool:
    """Whether a quarantined record is one a dialog may ask about."""
    return explain.code_of(record) in NEEDS_PERSON or record.reason.startswith(_DECOY_REASONS)


class HeadsUp:
    """Raises the dialog for what is quarantined and needs a person."""

    def __init__(self, state: Path, *, clock: Callable[[], float], store: Store,
                 bus: Bus | None = None, wires: Wires | None = None) -> None:
        self.state, self.clock, self.store, self.bus = Path(state), clock, store, bus
        self.choose, self.spawn, env = wires or Wires()
        self.env = os.environ if env is None else env
        self.lock = self.state.with_name("heads-up.lock")
        self._local = requests.Inbox(memory=True)

    def inbox(self) -> requests.Inbox:
        """The user's inbox, or this process's own when the keystore cannot be read here (a
        background process): the dialog still shows and answers what this process raised."""
        held = requests.default()
        return held if held.ready() else self._local

    # -- when to ask ---------------------------------------------------------------
    def on_quarantine(self, record: Record) -> None:
        """Store hook: ask about it unless it is a person's own doing or needs no person.
        Never raises."""
        try:
            if record.history and record.history[-1].get("actor") == "human":
                return
            if needs_person(record):
                self.spawn(self.prompt)
        except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
            logger.warning("sentinel heads-up failed: %s", type(exc).__name__)

    def poll(self) -> None:
        """Ask again about what is still pending once the cooldown is over. Never raises."""
        try:
            if self.enabled() and self._pending(self._load(), everything=False):
                self.spawn(self.prompt)
        except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
            logger.warning("sentinel heads-up failed: %s", type(exc).__name__)

    def enabled(self) -> bool:
        """Whether a dialog may be shown now: ``POOLHOUSE_NOTIFY`` is read on every call."""
        if self.choose is not None:
            return self.env.get(desktop.ENV, "system").strip().lower() != "off"
        return desktop.which_way(env=self.env) != "none"

    # -- the dialog ----------------------------------------------------------------
    def prompt(self) -> str:
        """Put the dialog up if nobody else has, the cooldown is over and something is
        pending; returns the label answered, or an empty string when nothing was shown."""
        return self._single(everything=False)

    def review(self) -> str:
        """The same dialog for everything held, now, for a person who asked to see it."""
        return self._single(everything=True)

    def _single(self, *, everything: bool) -> str:
        try:
            with only_one(self.lock, wait=False, note="heads-up"):
                return self._ask(everything)
        except Busy:
            return ""
        except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
            logger.warning("sentinel heads-up failed: %s", type(exc).__name__)
            return ""

    def _ask(self, everything: bool) -> str:
        now, memo, held = self.clock(), self._load(), self.inbox()
        self._settle(memo, now, held)
        rows = self._held(memo, everything=everything)
        if rows:
            self._raise(rows[:SHOWN_MOST], len(rows), held)
        if not self.enabled() or not self._open(memo, everything=everything):
            self._save(memo)
            return ""
        oldest = requests.oldest_pending(inbox=held)
        if oldest is None:
            self._save(memo)
            return ""
        memo["until"] = now + COOLDOWN_S
        self._save(memo)
        answer = dialog.ask(oldest, self.choose or self._desktop, env=self.env, inbox=held)
        again = requests.get(oldest.id, inbox=held)
        if answer == RELEASE and again is not None and again.state == "pending":
            self._emit("sentinel.release_refused", {"why": "the dialog's answer was not taken"})
        if answer in (LATER, "timeout"):
            memo["until"] = now + LATER_S
        self._settle(memo, now, held)
        self._save(memo)
        return str(answer)

    def _open(self, memo: dict[str, Any], *, everything: bool) -> bool:
        return everything or self.clock() >= memo["until"]

    def _raise(self, shown: list[Record], total: int, held: requests.Inbox) -> None:
        """Put what is held in front of the person as one request, replacing the last one."""
        title, body = self._compose([explain.describe(r, n, self.clock()) for n, r in enumerate(shown, 1)],
                                    total)
        ids = ",".join(r.id for r in shown)
        wanted = requests.Ask("quarantine_release", title, body, ("later", "keep-held", "release"),
                              requests.Origin("sentinel"), ttl=LATER_S, key="heads-up",
                              extra=(("ids", ids), ("total", str(total))))
        for each in requests.list_requests(state="pending", agent="sentinel", kind="quarantine_release",
                                            inbox=held):
            if each.extra.get("ids") == ids:
                return
        requests.raise_request(wanted, inbox=held)

    def _is_ours(self, each: requests.Request, ids: tuple[str, ...]) -> bool:
        """Whether ``each`` says exactly what sentinel would say about ``ids``: a request another
        component raised, or one whose text and ids disagree, releases nothing."""
        records = [self.store.get(i) for i in ids]
        if not ids or any(r is None for r in records):
            return False
        total = int(each.extra.get("total", "0") or 0) if each.extra.get("total", "").isdigit() else 0
        title, body = self._compose([explain.describe(r, n, self.clock()) for n, r in enumerate(records, 1)
                                     if r is not None], max(total, len(ids)))
        return (requests.model.shown(title, "subject"), requests.model.shown(body, "reason")) == \
            (each.subject, each.reason)

    def _settle(self, memo: dict[str, Any], now: float, held: requests.Inbox) -> None:
        """Act on the heads-up requests a person has answered, wherever they answered. A release
        is still the click: re-checked here, for exactly the ids the answered request listed."""
        for each in requests.list_requests(agent="sentinel", kind="quarantine_release", inbox=held):
            if each.state == "pending" or each.id in memo["settled"]:
                continue
            memo["settled"][each.id] = now
            ids = tuple(i for i in each.extra.get("ids", "").split(",") if i)
            if each.answer == "release" and each.state == "approved":
                if self._is_ours(each, ids):
                    self._release(ids, RELEASE)
                else:
                    self._emit("sentinel.release_refused", {"why": "the request is not what sentinel raised"})
            elif each.answer == "keep-held":
                memo["kept"].update(dict.fromkeys(ids, now))
            elif each.answer == "later":
                memo["until"] = max(memo["until"], now + LATER_S)
        memo["settled"] = dict(sorted(memo["settled"].items(), key=lambda kv: kv[1])[-50:])

    def _desktop(self, title: str, body: str, buttons: tuple[str, ...]) -> str:
        return desktop.choose(title, body, buttons, way=desktop.which_way(env=self.env))

    def _compose(self, items: list[explain.Item], total: int) -> tuple[str, str]:
        if total == 1:
            return f"Poolhouse is holding {items[0].name}", f"{items[0].why} {items[0].blocks}"
        names = "; ".join(f"{i.name} ({explain.short_code(i.code)})" for i in items)
        more = f" {total - len(items)} more are not listed and stay held." if total > len(items) else ""
        return (f"Poolhouse is holding {total} things",
                f"{names}.{more} Each is something that changed or was forged.")

    # -- the click -----------------------------------------------------------------
    def _release(self, ids: tuple[str, ...], answer: str) -> None:
        done: list[str] = []
        try:
            for ident in ids:
                record = self.store.get(ident)
                if record is None or record.state != State.QUARANTINED:
                    continue
                grant = human.mint_clicked("release", ident, answer=answer, label=RELEASE,
                                           env=self.env)
                self.store.release(ident, grant, "released by the dialog's Release button")
                done.append(ident)
        except human.HumanRequired as exc:
            self._emit("sentinel.release_refused", {"why": str(exc)[:120]})
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("sentinel heads-up release failed: %s", type(exc).__name__)
        if done:
            self._emit("sentinel.released_by_dialog", {"ids": done})

    def _emit(self, kind: str, evidence: dict[str, Any]) -> None:
        if self.bus is not None:
            self.bus.emit(Event(kind, Severity.NOTICE, "heads_up", "", evidence, self.clock()))

    # -- what is pending, and what was said ----------------------------------------
    def _pending(self, memo: dict[str, Any], *, everything: bool) -> list[Record]:
        if not everything and self.clock() < memo["until"]:
            return []
        return self._held(memo, everything=everything)

    def _held(self, memo: dict[str, Any], *, everything: bool) -> list[Record]:
        rows = [r for r in self.store.records(state=State.QUARANTINED)
                if everything or (needs_person(r) and r.id not in memo["kept"])]
        return sorted(rows, key=lambda r: (r.updated, r.id), reverse=True)

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.state.read_text())
        except (OSError, ValueError):
            data = {}
        until, kept = data.get("until", 0.0), data.get("kept", {})
        settled = data.get("settled", {})
        return {"version": 3, "until": until if isinstance(until, int | float) else 0.0,
                "kept": {k: t for k, t in kept.items() if isinstance(k, str)}
                if isinstance(kept, dict) else {},
                "settled": {k: t for k, t in settled.items() if isinstance(k, str)}
                if isinstance(settled, dict) else {}}

    def _save(self, memo: dict[str, Any]) -> None:
        write_text(self.state, json.dumps(memo, sort_keys=True))
