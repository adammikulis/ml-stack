"""Web components over the line limit."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._util import read, rel

NAME = "deep-components"
OWNER = ""
ROOTS = ("src/ml_stack",)
SUFFIXES = (".html", ".js", ".css")
LIMIT = 500


def describe() -> str:
    return (f"A component over {LIMIT} lines; it holds more than one screen. "
            "Split it and name each part in the page that assembles them.")


def find(root: Path) -> list[Finding]:
    out = []
    for where in ROOTS:
        for path in sorted((root / where).rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            lines = len(read(path).splitlines())
            if lines > LIMIT:
                out.append(Finding(rel(path, root), 1, f"{lines} lines"))
    return out
