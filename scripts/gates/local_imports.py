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


def _modules(node: ast.stmt) -> list[str]:
    """The ml_stack modules one import statement defers, empty for any other statement."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names if a.name.split(".")[0] == "ml_stack"]
    if isinstance(node, ast.ImportFrom):
        if node.level:
            return ["." * node.level + (node.module or "")]
        if (node.module or "").split(".")[0] == "ml_stack":
            return [node.module or ""]
    return []


def _imports(body: list[ast.stmt]) -> list[tuple[str, ast.stmt]]:
    """Each ml_stack module this body defers, once, with the statement that named it.

    A module named twice in one function is one deferred dependency however many statements
    say so, so splitting ``from x import a, b as c`` in two does not change the count.
    """
    found: list[tuple[str, ast.stmt]] = []
    seen: set[str] = set()
    for statement in body:
        for node in ast.walk(statement):
            for module in _modules(node):
                if module in seen:
                    continue
                seen.add(module)
                found.append((module, node))
    return found


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        tree = parse(path)
        if tree is None:
            continue
        seen: set[tuple[str, str]] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for module, found in _imports(node.body):
                if (node.name, module) in seen:
                    continue
                seen.add((node.name, module))
                out.append(Finding(where, found.lineno, f"{module} in {node.name}()"))
    return out
