"""Functions over the statement limit."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._util import parse, python_files, rel

NAME = "long-functions"
OWNER = ""
ROOTS = ("src/ml_stack",)
LIMIT = 80


def describe() -> str:
    return f"A function over {LIMIT} statements; split it where it changes subject."


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        tree = parse(path)
        if tree is None:
            continue
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
