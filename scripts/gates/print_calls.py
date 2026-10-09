"""print() in library code, outside the modules a console script runs."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import calls, dotted, exempt, parse

NAME = "print-calls"
OWNER = "ml_stack.log"
INCREMENTAL = True
ROOTS = ("src/ml_stack",)
OWNS = ("src/ml_stack/log.py",)

COMMANDS: set[str] = set()
"""Command modules still printing straight to the console."""


def describe() -> str:
    return "A print() in library code; `ml_stack.log` says it, so a caller can redirect it."


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    if where in OWNS or where in COMMANDS or exempt(where, tuple(COMMANDS)):
        return out
    tree = parse(path)
    if tree is None:
        return out
    for node, name in calls(tree):
        if name == "print" and _to_the_console(node):
            out.append(Finding(where, node.lineno, "print()"))
    return out


find = finder(ROOTS, scan)


def _to_the_console(node: ast.Call) -> bool:
    """Whether this print writes to stdout or stderr rather than a stream it was handed."""
    for keyword in node.keywords:
        if keyword.arg == "file":
            return dotted(keyword.value) in ("sys.stdout", "sys.stderr", "stdout", "stderr")
    return True
