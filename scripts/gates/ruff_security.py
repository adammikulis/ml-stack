"""Ruff's S rules: the bandit checks."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._ruff import findings, skip  # noqa: F401

NAME = "ruff-security"
OWNER = ""
PREFIXES = frozenset({"S"})


def describe() -> str:
    return "A bandit finding: a swallowed exception, an unvalidated URL, a binding to every address."


def find(root: Path) -> list[Finding]:
    return findings(root, PREFIXES)
