"""The processes above this one: which harness process a hook or a guard runs under."""

from __future__ import annotations

import os

import psutil

__all__ = ["HARNESS_NAMES", "ancestors", "harness_parent", "under_harness"]

SHELLS = {"sh", "bash", "zsh", "dash", "fish", "ksh", "env", "sudo", "login"}
HARNESS_NAMES = {"claude", "codex"}
DEPTH = 12


def ancestors(pid: int | None = None) -> list[tuple[int, float, str]]:
    """``(pid, create_time, name)`` of each ancestor of ``pid`` (default this process), nearest first."""
    out: list[tuple[int, float, str]] = []
    try:
        process = psutil.Process(os.getpid() if pid is None else pid).parent()
        while process is not None and process.pid > 1 and len(out) < DEPTH:
            out.append((process.pid, process.create_time(), process.name().lower()))
            process = process.parent()
    except (psutil.Error, OSError):
        pass
    return out


def harness_parent() -> tuple[int, float] | None:
    """The nearest ancestor that is not a shell: the process that ran this hook for the harness."""
    for pid, created, name in ancestors():
        if name not in SHELLS:
            return pid, created
    return None


def under_harness(bound: set[tuple[int, float]] | frozenset[tuple[int, float]] = frozenset()) -> bool:
    """Whether an ancestor is a known harness process or one a hook recorded for a session."""
    return any(name in HARNESS_NAMES or name.startswith(("claude", "codex")) or (pid, created) in bound
               for pid, created, name in ancestors())
