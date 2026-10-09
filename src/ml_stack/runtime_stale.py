"""Processes that started before the selected runtime was verified, and so still run older code.

`ml-stack runtime ensure` switches what the next start runs; it signals nothing, because a store
writer is never killed. A daemon started before the switch keeps running the build it started
with (the board daemon once ran an old build for hours). This names those processes, so
`runtime status` can say so.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from ml_stack.serve import process

__all__ = ["Running", "older_than", "running", "stale_lines"]

MARKERS = ("ml_stack", "ml-stack")
"""A process is one of ours when a word of its command line holds one of these."""
SLACK_S = 5.0
"""A process that started within this long of the verification is not older than it."""


class Running(NamedTuple):
    """One process: its id, when it started (epoch seconds) and its command line."""

    pid: int
    started: float
    argv: tuple[str, ...]


def running() -> list[Running]:
    """Every readable process of ours (a command-line word names ml_stack or an ml-stack program), this one excluded."""
    return [Running(*row) for row in process.command_lines() if any(m in word for word in row[2] for m in MARKERS)]


def older_than(procs: list[Running], prefix: Path, verified_at: float) -> list[Running]:
    """The processes that began before ``verified_at`` and do not run from ``prefix`` (the selected runtime).

    A host that replaced itself with `execve` keeps its start time but its command line names the
    selected runtime's interpreter, so it is not listed.
    """
    if verified_at <= 0:
        return []
    inside = str(prefix)
    return [p for p in procs if p.started + SLACK_S < verified_at and not any(inside in word for word in p.argv)]


def stale_lines(prefix: Path, verified_at: float, procs: list[Running] | None = None) -> list[str]:
    """One status line per process older than the selected runtime, then the line that says what to do."""
    old = older_than(running() if procs is None else procs, prefix, verified_at)
    lines = [f"older        pid {p.pid} began {int(verified_at - p.started)}s before the selected runtime: {' '.join([Path(p.argv[0]).name, *p.argv[1:]])[:100]}"
             for p in old]
    if old:
        lines.append(f"note         {len(old)} process(es) run code older than the selected runtime; "
                     "stop them through their own control (never a kill) and start them again")
    return lines
