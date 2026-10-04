"""The project a connection is for: the memory package's detection, with a display name that is
safe to show to a model."""

from __future__ import annotations

import re
from pathlib import Path

from ml_stack.memory.project import detect

__all__ = ["clean", "describe"]

NAME_MAX = 40


def clean(text: str) -> str:
    """``text`` cut to letters, digits, dots, dashes and underscores, at most 40 characters."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_.")[:NAME_MAX]


def describe(path: str = "", *, none: bool = False, start: Path | None = None) -> dict[str, str]:
    """``{"key", "name"}`` of the project for ``path``, else the folder ``start`` (default the
    working directory) sits in; empty for ``none``, the home folder or the filesystem root."""
    if none:
        return {}
    found = detect(start, explicit=Path(path) if path else None)
    return {} if found is None else {"key": found.key, "name": clean(found.name) or "project"}
