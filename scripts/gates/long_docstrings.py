"""Docstrings over the line limit."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._util import parse, python_files, rel

NAME = "long-docstrings"
OWNER = ""
ROOTS = ("src/ml_stack",)
LIMIT = 12
HOLDERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def describe() -> str:
    return f"A docstring over {LIMIT} lines; say what it returns and put the rest in a commit."


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        tree = parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, HOLDERS):
                continue
            text = ast.get_docstring(node, clean=False)
            if not text:
                continue
            lines = len(text.splitlines())
            if lines > LIMIT:
                name = getattr(node, "name", where)
                out.append(Finding(where, node.body[0].lineno, f"{name}: {lines} lines"))
    return out
