"""Files over the line limit."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._util import python_files, read, rel

NAME = "deep-files"
OWNER = ""
ROOTS = ("src/ml_stack",)
LIMIT = 900


def describe() -> str:
    return f"A file over {LIMIT} lines; it holds more than one job."


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        lines = len(read(path).splitlines())
        if lines > LIMIT:
            out.append(Finding(rel(path, root), 1, f"{lines} lines"))
    return out
