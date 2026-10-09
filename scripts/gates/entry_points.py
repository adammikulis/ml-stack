"""main() definitions across the library."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import parse

NAME = "entry-points"
OWNER = "ml_stack.cli"
INCREMENTAL = True
ROOTS = ("src/ml_stack",)


def describe() -> str:
    return "Another main(); a command reaches the library through ml_stack.cli."


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    tree = parse(path)
    if tree is None:
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main":
            out.append(Finding(where, node.lineno, "def main"))
    return out


find = finder(ROOTS, scan)
