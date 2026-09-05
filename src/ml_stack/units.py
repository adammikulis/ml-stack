"""How a number is written for a person to read."""

from __future__ import annotations

__all__ = ["human_bytes"]


def human_bytes(size: float) -> str:
    """``1536`` -> ``1.5K``, ``2**30`` -> ``1.0G``. Bytes are whole, everything else 1dp."""
    value = float(size)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}T"
