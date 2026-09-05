"""Ruff's B rules: shapes that are usually a bug rather than a style."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._ruff import findings, skip  # noqa: F401

NAME = "ruff-bugbear"
OWNER = ""
PREFIXES = frozenset({"B"})


def describe() -> str:
    return "A bugbear shape: a mutable default, a loop variable closed over, a zip without strict."


def find(root: Path) -> list[Finding]:
    return findings(root, PREFIXES)
