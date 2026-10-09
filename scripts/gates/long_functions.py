"""Functions over the statement limit."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import parse

NAME = "long-functions"
OWNER = ""
INCREMENTAL = True
ROOTS = ("src/ml_stack",)
LIMIT = 80


def describe() -> str:
    return f"A function over {LIMIT} statements; split it where it changes subject."


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    tree = parse(path)
    if tree is None:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        count = sum(
            1
            for statement in node.body
            for inner in ast.walk(statement)
            if isinstance(inner, ast.stmt)
        )
        if count > LIMIT:
            out.append(Finding(where, node.lineno, f"{node.name}: {count} statements"))
    return out


find = finder(ROOTS, scan)
