"""How a number is written for a person to read."""

from __future__ import annotations

__all__ = ["human_bytes", "span"]


def human_bytes(size: float) -> str:
    """``1536`` -> ``1.5K``, ``2**30`` -> ``1.0G``. Bytes are whole, everything else 1dp."""
    value = float(size)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}T"


def span(seconds: float) -> str:
    """``45 s``, ``26 min``, ``2 h 10 min`` -- the shapes `history.parse_duration` reads."""
    whole = int(round(seconds))
    if whole < 60:
        return f"{whole} s"
    minutes = int(round(whole / 60))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"
