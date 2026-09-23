"""How a number is written for a person to read, and read back."""

from __future__ import annotations

import re

__all__ = ["human_bytes", "parse_duration", "span"]

_DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hours?|m|min|mins|minutes?|s|sec|secs|seconds?)\b")


def human_bytes(size: float) -> str:
    """``1536`` -> ``1.5K``, ``2**30`` -> ``1.0G``. Bytes are whole, everything else 1dp."""
    value = float(size)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}T"


def span(seconds: float) -> str:
    """``45 s``, ``26 min``, ``2 h 10 min`` -- the shapes `parse_duration` reads."""
    whole = round(seconds)
    if whole < 60:
        return f"{whole} s"
    minutes = round(whole / 60)
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def parse_duration(text: str) -> float | None:
    """``2h 15m``, ``90s``, ``1.5h``, ``~600`` -> seconds; a bare number is seconds."""
    total, found = 0.0, False
    for amount, unit in _DURATION.findall(text):
        found = True
        total += float(amount) * {"h": 3600.0, "m": 60.0, "s": 1.0}[unit[0]]
    if found:
        return total
    bare = re.search(r"\d+(?:\.\d+)?", text)
    return float(bare.group()) if bare else None
