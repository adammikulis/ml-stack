"""print() in library code, outside the modules a console script runs."""

from __future__ import annotations

from pathlib import Path

import ast

from . import Finding
from ._util import calls, dotted, exempt, parse, python_files, rel

NAME = "print-calls"
OWNER = "ml_stack.log"
ROOTS = ("src/ml_stack",)
OWNS = ("src/ml_stack/log.py",)

COMMANDS = {
    "src/ml_stack/bench/run.py",
}
"""Command modules still printing straight to the console."""


def describe() -> str:
    return "A print() in library code; `ml_stack.log` says it, so a caller can redirect it."


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        if where in OWNS or where in COMMANDS or exempt(where, tuple(COMMANDS)):
            continue
        tree = parse(path)
        if tree is None:
            continue
        for node, name in calls(tree):
            if name == "print" and _to_the_console(node):
                out.append(Finding(where, node.lineno, "print()"))
    return out


def _to_the_console(node: ast.Call) -> bool:
    """Whether this print writes to stdout or stderr rather than a stream it was handed."""
    for keyword in node.keywords:
        if keyword.arg == "file":
            return dotted(keyword.value) in ("sys.stdout", "sys.stderr", "stdout", "stderr")
    return True
