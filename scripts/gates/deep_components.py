"""Web components over the line limit."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._perfile import each_file
from ._util import read

NAME = "deep-components"
OWNER = ""
INCREMENTAL = True
ROOTS = ("src/ml_stack",)
SUFFIXES = (".html", ".js", ".css")
LIMIT = 500
HARD = True


def describe() -> str:
    return (f"A component over {LIMIT} lines; it holds more than one screen. "
            "Split it and name each part in the page that assembles them. "
            "There is no allowance: the limit is the limit.")


def scan(path: Path, where: str) -> list[Finding]:
    lines = len(read(path).splitlines())
    return [Finding(where, 1, f"{lines} lines")] if lines > LIMIT else []


def find(root: Path) -> list[Finding]:
    paths = [path for where in ROOTS for path in sorted((root / where).rglob("*"))
             if path.suffix in SUFFIXES and path.is_file()]
    return each_file(root, paths, scan)
