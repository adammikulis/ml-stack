"""ml_stack imports inside a function body."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._util import parse, python_files, rel

NAME = "local-imports"
OWNER = ""
ROOTS = ("src/ml_stack",)


def describe() -> str:
    return "An ml_stack import inside a function; a deferred import hides an import cycle."


def _imports(body: list[ast.stmt]) -> list[ast.stmt]:
    found = []
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] == "ml_stack" for a in node.names):
                    found.append(node)
            elif isinstance(node, ast.ImportFrom):
                if node.level or (node.module or "").split(".")[0] == "ml_stack":
                    found.append(node)
    return found


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        tree = parse(path)
        if tree is None:
            continue
        seen: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for found in _imports(node.body):
                if id(found) in seen:
                    continue
                seen.add(id(found))
                out.append(Finding(where, found.lineno, f"in {node.name}()"))
    return out
