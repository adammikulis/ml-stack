"""Docstrings over the line limit."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import parse

NAME = "long-docstrings"
OWNER = ""
INCREMENTAL = True
ROOTS = ("src/poolhouse",)
LIMIT = 12
HOLDERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def describe() -> str:
    return f"A docstring over {LIMIT} lines; say what it returns and put the rest in a commit."


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    tree = parse(path)
    if tree is None:
        return out
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


find = finder(ROOTS, scan)
