"""Files over the line limit."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import read

NAME = "deep-files"
OWNER = ""
INCREMENTAL = True
ROOTS = ("src/poolhouse",)
LIMIT = 900
HARD = True


def describe() -> str:
    return (f"A file over {LIMIT} lines; it holds more than one job. "
            "Split it into a module per job. There is no allowance: the limit is the limit.")


def scan(path: Path, where: str) -> list[Finding]:
    lines = len(read(path).splitlines())
    return [Finding(where, 1, f"{lines} lines")] if lines > LIMIT else []


find = finder(ROOTS, scan)
