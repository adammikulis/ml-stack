"""Who wrote under the real state root while the session ran, and what that means for the run.

The rule (``tests/README.md``, "Writes under the real state root"): a changed file that no
existing attribution explains and that is not a zero-byte ``*.lock`` is a failure when no other
live poolhouse process was seen at the start or the end of the run, and a warning, naming the
files and the writers, when one was.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable
from pathlib import Path

import psutil

Row = tuple[int, str]  # pid, command line
FIXED_WRITERS = "POOLHOUSE_TEST_OTHER_WRITERS"


def _is_poolhouse(argv: list[str]) -> bool:
    """A process started as a poolhouse console script or as ``python -m poolhouse...``."""
    heads = [Path(a).name for a in argv[:3]]
    if any(h.startswith("poolhouse") for h in heads):
        return True
    return any(a == "-m" and i + 1 < len(argv) and argv[i + 1].startswith("poolhouse")
               for i, a in enumerate(argv))


def session_tree() -> set[int]:
    """This process, its ancestors that are part of the same test run, and their descendants."""
    me = psutil.Process()
    top = me
    for parent in me.parents():
        try:
            line = " ".join(parent.cmdline())
        except psutil.Error:
            break
        if not any(word in line for word in ("pytest", "scripts/test", "testslots")):
            break
        top = parent
    try:
        return {top.pid, me.pid, *(p.pid for p in top.children(recursive=True))}
    except psutil.Error:
        return {me.pid}


def live_poolhouse_processes() -> list[Row]:
    """Every live poolhouse process outside this test session; a test fixes the answer by
    putting a JSON list of ``[pid, command line]`` in ``POOLHOUSE_TEST_OTHER_WRITERS``."""
    if (fixed := os.environ.get(FIXED_WRITERS)) is not None:
        return [(int(pid), str(line)) for pid, line in json.loads(fixed)]
    ours = session_tree()
    rows = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        argv = process.info.get("cmdline") or []
        if process.info["pid"] not in ours and _is_poolhouse(argv):
            rows.append((process.info["pid"], " ".join(argv)[:120]))
    return rows


def _empty_lock(root: Path, name: str) -> bool:
    if not name.endswith(".lock"):
        return False
    try:
        return (root / name).lstat().st_size == 0
    except OSError:
        return True  # gone: a lock released


class Watch:
    """Snapshot ``root`` and the other live writers now; ``settle`` judges at the end."""

    def __init__(self, root: Path, mtimes: Callable[[Path], dict[str, int]],
                 changes: Callable[[dict, dict], list[str]],
                 processes: Callable[[], list[Row]] = live_poolhouse_processes) -> None:
        self.root, self.mtimes, self.changes, self.processes = root, mtimes, changes, processes
        self.writers = dict(self.processes())
        self.before = self.mtimes(root)

    def verdict(self) -> tuple[list[str], list[str], dict[int, str]]:
        """``(failures, warnings, writers)`` for what changed since construction."""
        writers = {**self.writers, **dict(self.processes())}
        written = [n for n in self.changes(self.before, self.mtimes(self.root))
                   if not _empty_lock(self.root, n)]
        return ([], written, writers) if writers and written else (written, [], writers)

    def settle(self) -> str:
        """Warn once for a possibly external write; return the failure message, or ``""``."""
        failed, possible, writers = self.verdict()
        if possible:
            seen = "; ".join(f"{pid} {line}" for pid, line in sorted(writers.items())[:5])
            warnings.warn(f"the real state root {self.root} changed during the run, with other "
                          f"poolhouse processes live ({seen}): " + ", ".join(possible[:20]),
                          stacklevel=2)
        if failed:
            return (f"the real state root {self.root} was written during the run: "
                    + ", ".join(failed[:20]))
        return ""
