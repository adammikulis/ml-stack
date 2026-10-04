"""Plain, bounded one-line text for anything read from the workspace and shown or listed."""

from __future__ import annotations

import re

__all__ = ["line", "name_ok", "text"]

HIDDEN = re.compile(r"[\x00-\x1f\x7f-\x9f\u061c\u200b-\u200f\u2028-\u202e\u2060-\u206f\ufeff]")
BOARD = re.compile(r"^#[a-z0-9][a-z0-9._-]{0,39}$")


def line(value: object, width: int = 120) -> str:
    """``value`` on one line: control and bidirectional characters turned to spaces, at most
    ``width`` characters."""
    return " ".join(HIDDEN.sub(" ", str(value)).split())[:width]


def text(value: object, width: int = 4000) -> tuple[str, bool]:
    """``value`` with control and bidirectional characters removed (newlines kept), cut to
    ``width``; whether it was cut."""
    kept = HIDDEN.sub(lambda m: "\n" if m.group() == "\n" else " ", str(value).replace("\r", ""))
    return kept[:width], len(kept) > width


def name_ok(name: str) -> bool:
    """Whether ``name`` is a board name: ``#`` then lowercase letters, digits, ``.``, ``_``, ``-``."""
    return bool(BOARD.match(name)) and ".." not in name
