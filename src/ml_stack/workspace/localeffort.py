"""Effort: how much a local agent's model thinks and may write per reply. It is compute only:
it never changes the role, the tools or the caps on steps, calls and seconds."""

from __future__ import annotations

import re

__all__ = ["AUTO", "DEFAULT", "DEFAULT_MAX", "LEVELS", "TOKENS", "clamp", "pick_for", "thinks", "valid"]

LEVELS = ("off", "low", "medium", "high")
AUTO = "auto"
DEFAULT = "off"
DEFAULT_MAX = "medium"
TOKENS = {"off": 2048, "low": 4096, "medium": 8192, "high": 16384}
"""The most tokens one reply may hold at each level."""
_HARD = re.compile(r"\b(plan|diagnos\w*|debug\w*|review|design|investigat\w*|why)\b", re.I)
_EASY = re.compile(r"\b(status|list|lookup|look up|show|what is|which|count)\b", re.I)


def valid(level: str, *, allow_auto: bool = False) -> str:
    """``level`` when it is a level (or ``auto`` where allowed); ValueError otherwise."""
    if level in LEVELS or (allow_auto and level == AUTO):
        return level
    raise ValueError(f"effort is one of {', '.join(LEVELS)}{', auto' if allow_auto else ''}")


def clamp(level: str, ceiling: str) -> str:
    """``level``, or ``ceiling`` when ``level`` is above it."""
    return level if LEVELS.index(level) <= LEVELS.index(ceiling) else ceiling


def thinks(level: str) -> bool:
    return level != "off"


def pick_for(task: str, ceiling: str) -> str:
    """The level the rule table gives a task, never above ``ceiling``: a plan, diagnosis, review or
    design is medium, a status or lookup is off, anything else is low."""
    if _EASY.search(task) and not _HARD.search(task):
        return "off"
    return clamp("medium" if _HARD.search(task) else "low", ceiling)
