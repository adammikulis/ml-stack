"""Every ruff rule that is not BLE, B or S."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._ruff import findings, skip  # noqa: F401
from .ruff_blind_except import PREFIXES as BLE
from .ruff_bugbear import PREFIXES as BUGBEAR
from .ruff_security import PREFIXES as SECURITY

NAME = "ruff-other"
OWNER = ""
PREFIXES = BLE | BUGBEAR | SECURITY


def describe() -> str:
    return "A ruff finding: unsorted imports, a wide signature, syntax older than the minimum Python."


def find(root: Path) -> list[Finding]:
    return findings(root, PREFIXES, invert=True)
