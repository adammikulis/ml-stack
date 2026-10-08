"""Telling a write under the real state root that another agent's process made from one a test made."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import psutil


def _served_roots() -> list[tuple[str, int]]:
    """``(root, pid)`` for every live process started with ``--root DIR``."""
    found = []
    try:
        for process in psutil.process_iter():
            try:
                argv = process.cmdline()
                if "--root" in argv[:-1]:
                    found.append((argv[argv.index("--root") + 1], process.pid))
            except (psutil.Error, OSError):
                continue
    except (AttributeError, TypeError):  # a test replaced psutil's process type
        return []
    return found


def served_by_foreign(root: Path, rel: Path, ours: Callable[[dict], bool]) -> bool:
    """Whether ``rel`` is below a directory a live process outside this session serves with
    ``--root`` (the board daemon at ``traind``, a task daemon, any other agent's server)."""
    top = str(root / rel.parts[0]) if rel.parts else ""
    return bool(top) and any(where == top and not ours({"owner_pid": pid})
                             for where, pid in _served_roots())


def held_by_foreign(root: Path, rels: list[Path]) -> set[Path]:
    """The ``rels`` some process outside this session has open right now."""
    wanted = {str(root / rel): rel for rel in rels}
    held: set[Path] = set()
    try:
        me = psutil.Process(os.getpid())
        mine = {me.pid, *(p.pid for p in me.parents()), *(p.pid for p in me.children(True))}
    except (psutil.Error, OSError, TypeError, AttributeError):
        return held
    try:
        for process in psutil.process_iter():
            try:
                if process.pid in mine:
                    continue
                held.update(wanted[f.path] for f in process.open_files() if f.path in wanted)
            except (psutil.Error, OSError):
                continue
    except (AttributeError, TypeError):  # a test replaced psutil's process type
        return held
    return held
