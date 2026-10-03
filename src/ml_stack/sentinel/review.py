"""``ml-stack-security review``: everything sentinel holds on one screen, and the keys to deal
with it, for a person at a real terminal.

Viewing is not privileged: `held_items` and `table` work anywhere, for an agent too, and only
ever print what `explain` has made safe. Acting is: the interactive screen refuses unless a
person is at a terminal and no agent marker is set (the same rule as ``release``), a release
takes one confirming key and shows what it will unblock, and a purge still needs the id typed
in full. All keys are read from the terminal itself; nothing in a record, a file or the
environment can press one.
"""

from __future__ import annotations

import os
import select
import shutil
import sys
from collections.abc import Callable
from typing import Any, TextIO

from ml_stack import sentinel
from ml_stack.sentinel import explain, human
from ml_stack.sentinel.explain import Item, show
from ml_stack.sentinel.store import Record, State, Store, TransitionRefused

__all__ = ["NeedsTerminal", "Review", "held_items", "hint", "interactive", "table"]

CLEAR = "\x1b[2J\x1b[H"
KEYS = "1-9 select   r release   k keep   d details   R release all watched   p purge   q quit"
COMMAND = "ml-stack-security review"
COMMAND_SHOW = "ml-stack-security quarantine show"


class NeedsTerminal(Exception):
    """The interactive screen cannot run here, with the reason."""


def held_items(store: Store, now: float) -> list[Item]:
    """Every quarantined and watched subject, newest first, numbered from 1."""
    rows = [r for r in store.records() if r.state in (State.QUARANTINED, State.WATCH)]
    rows.sort(key=lambda r: (r.updated, r.created, r.id), reverse=True)
    return [explain.describe(r, n, now) for n, r in enumerate(rows, 1)]


def hint(items: list[Item]) -> str:
    """``2 held: peer 127.0.0.1 (forged traffic), server :51089. Review them: ...``, or
    an empty string when nothing is held."""
    if not items:
        return ""
    names = ", ".join(f"{i.name}" + (f" ({explain.short_code(i.code)})" if i.code else "")
                      for i in items[:3])
    more = f" and {len(items) - 3} more" if len(items) > 3 else ""
    return f"{len(items)} held: {names}{more}. Review them: {COMMAND}"


def table(items: list[Item]) -> list[str]:
    """The listing ``review --list`` prints: four lines per subject."""
    if not items:
        return ["nothing is held"]
    out: list[str] = []
    for i in items:
        out += [f"{i.number:>2}  {i.state.upper():<11} {i.name}   {i.age}",
                f"    why:    {i.why}", f"    blocks: {i.blocks}", f"    next:   {i.advice}"]
    return out


class Review:
    """The interactive screen over ``store``. ``key`` returns one key press, ``typed`` reads
    a line; both are wired to the real terminal by `interactive`."""

    def __init__(self, node: sentinel.Sentinel, out: TextIO, key: Callable[[], str],
                 typed: Callable[[str], str]) -> None:
        self.node, self.out, self.key, self.typed = node, out, key, typed
        self.at, self.note = 0, ""

    # -- drawing ---------------------------------------------------------------------
    def items(self) -> list[Item]:
        return held_items(self.node.store, self.node.clock())

    def say(self, *lines: str) -> None:
        self.out.write("\n".join(lines) + "\n")
        self.out.flush()

    def draw(self) -> list[Item]:
        items = self.items()
        self.at = min(self.at, max(0, len(items) - 1))
        width = shutil.get_terminal_size((100, 24)).columns
        quarantined = sum(i.state == "quarantined" for i in items)
        head = f"ml-stack-security review   {quarantined} held, {len(items) - quarantined} watched"
        lines = [CLEAR + head, ""]
        for n, i in enumerate(items):
            mark = ">" if n == self.at else " "
            lines.append(f"{mark}{i.number:>2}  {i.state.upper():<11} {i.name}   {i.age}"[:width])
        if items:
            now = items[self.at]
            lines += ["", f"  why:    {now.why}", f"  blocks: {now.blocks}",
                      f"  next:   {now.advice}"]
        else:
            lines += ["", "  nothing is held"]
        lines += ["", self.note, KEYS]
        self.note = ""
        self.say(*lines)
        return items

    # -- the loop --------------------------------------------------------------------
    def run(self) -> int:
        while True:
            items = self.draw()
            pressed = self.key()
            if pressed in ("q", "\x03", "\x04", ""):
                return 0
            self.handle(pressed, items)

    def handle(self, pressed: str, items: list[Item]) -> None:
        moves = {"down": 1, "up": -1}
        if pressed in moves and items:
            self.at = (self.at + moves[pressed]) % len(items)
        elif pressed.isdigit() and 0 < int(pressed) <= len(items):
            self.at = int(pressed) - 1
        elif pressed == "k":
            self.note = "kept: nothing changed"
            self.at += 1
        elif pressed in ("r", "d", "p") and items:
            {"r": self.release, "d": self.details, "p": self.purge}[pressed](items[self.at])
        elif pressed == "R":
            self.release_watched(items)

    # -- acting ----------------------------------------------------------------------
    def release(self, item: Item) -> None:
        unblocks = "It stops being watched." if item.state == "watch" else (
            f"Releasing ends this: {item.blocks}")
        self.say("", f"Release {item.name}? {unblocks}   [y = yes, any other key = no]")
        self.note = self.try_release(item, lambda _: self.key() == "y")

    def try_release(self, item: Item, pressed: Callable[[str], bool]) -> str:
        try:
            grant = human.mint_pressed("release", item.id, pressed=pressed)
            self.node.store.release(item.id, grant)
        except (human.HumanRequired, TransitionRefused, KeyError, OSError) as exc:
            return f"not released: {show(exc, 100)}"
        return f"released {item.name}"

    def release_watched(self, items: list[Item]) -> None:
        watched = [i for i in items if i.state == "watch"]
        if not watched:
            self.note = "nothing is only watched"
            return
        names = ", ".join(i.name for i in watched[:4]) + ("..." if len(watched) > 4 else "")
        self.say("", f"Stop watching all {len(watched)} watched ({names})? Nothing quarantined "
                     "is touched.   [y = yes, any other key = no]")
        yes = self.key() == "y"
        done = [self.try_release(i, lambda _: yes) for i in watched]
        self.note = f"{sum(d.startswith('released') for d in done)} of {len(watched)} released"

    def purge(self, item: Item) -> None:
        self.say("", f"Purging deletes what is held for {item.name}; the record stays.")
        try:
            grant = human.mint("purge", item.id, typed=self.typed)
            self.node.store.purge(item.id, grant)
        except (human.HumanRequired, KeyError, OSError) as exc:
            self.note = f"not purged: {show(exc, 100)}"
            return
        self.note = f"purged {item.name}"

    def details(self, item: Item) -> None:
        record = self.node.store.get(item.id)
        self.say(CLEAR + f"{item.name}   ({item.id})", *self.record_lines(record), "",
                 "any key to go back")
        self.key()

    def record_lines(self, record: Record | None) -> list[str]:
        if record is None:
            return ["  the record is gone"]
        subject = f"{record.kind}:{record.key}"
        out = [f"  state:   {record.state.value}", f"  subject: {show(subject, 200)}",
               f"  reason:  {show(record.reason, 300)}"]
        out += [f"  {show(k, 30)}: {show(v, 120)}" for k, v in list(record.evidence.items())[:12]]
        out += ["", "  history:"]
        out += [f"    {show(h.get('from'), 12)} -> {show(h.get('to'), 12)} by "
                f"{show(h.get('actor'), 20)}: {show(h.get('reason'), 100)}"
                for h in record.history[-10:]]
        out += ["", "  events:", *self.event_lines(subject)]
        out += ["", f"  the same, by command: {COMMAND_SHOW} {record.id}",
                f"  release: ml-stack-security quarantine release {record.id}",
                f"  purge:   ml-stack-security quarantine purge {record.id}"]
        return out

    def event_lines(self, subject: str) -> list[str]:
        log = self.node.bus.log
        rows: list[Any] = [e for e in (log.read() if log else []) if e.subject == subject]
        return [f"    {e.ts:.0f} {e.severity.name.lower():<8} {show(e.kind, 40)}"
                for e in rows[-10:]] or ["    none"]


def _read_key(fd: int) -> str:
    """One key from a terminal in cbreak mode: a character, or ``up`` / ``down``."""
    first = os.read(fd, 1)
    if first != b"\x1b":
        return first.decode("ascii", errors="ignore")  # "" is end of input
    rest = b""
    while select.select([fd], [], [], 0.05)[0] and len(rest) < 2:
        rest += os.read(fd, 1)
    return {b"[A": "up", b"[B": "down"}.get(rest, "esc")


def interactive(node: sentinel.Sentinel | None = None) -> int:
    """Run the screen on this process's terminal. Raises `human.HumanRequired` when a person
    is not at it and `NeedsTerminal` where the platform has no terminal control."""
    human.require_person("review")
    try:
        import termios
        import tty
    except ImportError:
        raise NeedsTerminal("the interactive review needs a POSIX terminal; on this system "
                            f"use `{COMMAND} --list` and `ml-stack-security quarantine "
                            "release ID`") from None
    node = node or sentinel.default()
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)

    def typed(prompt: str) -> str:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        try:
            return input(prompt)
        finally:
            tty.setcbreak(fd)

    try:
        tty.setcbreak(fd)
        return Review(node, sys.stdout, lambda: _read_key(fd), typed).run()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
