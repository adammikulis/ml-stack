"""main() definitions across the library."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._util import parse, python_files, rel

NAME = "entry-points"
OWNER = "ml_stack.cli"
ROOTS = ("src/ml_stack",)


def describe() -> str:
    return "Another main(); a command reaches the library through ml_stack.cli."


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        tree = parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main":
                out.append(Finding(where, node.lineno, "def main"))
    return out
