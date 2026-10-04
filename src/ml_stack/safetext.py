"""Text from outside as one safe printable line."""

from __future__ import annotations

import unicodedata

__all__ = ["escape"]

_EDGE = "…"


def escape(text: str, limit: int = 60) -> str:
    """``text`` as one line of at most ``limit`` characters: every control, format (bidi,
    zero-width), separator, private or unassigned character written as a visible escape such
    as ``\\x1b`` or ``\\u202e``, cut with an ellipsis when long."""
    out = "".join(ch if _plain(ch) else _escape(ch) for ch in text)
    return out if len(out) <= limit else out[:limit - 1] + _EDGE


def _plain(ch: str) -> bool:
    return ch == " " or (ch.isprintable() and unicodedata.category(ch)[0] not in "CZ")


def _escape(ch: str) -> str:
    named = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
    if ch in named:
        return named[ch]
    point = ord(ch)
    if point < 0x100:
        return f"\\x{point:02x}"
    return f"\\u{point:04x}" if point < 0x10000 else f"\\U{point:08x}"
