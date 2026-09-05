"""Ruff's BLE rules: an except that catches everything."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._ruff import findings, skip  # noqa: F401

NAME = "ruff-blind-except"
OWNER = ""
PREFIXES = frozenset({"BLE"})


def describe() -> str:
    return "An except that catches every exception; name the ones this call raises."


def find(root: Path) -> list[Finding]:
    return findings(root, PREFIXES)
