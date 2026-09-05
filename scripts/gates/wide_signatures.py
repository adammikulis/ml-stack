"""Functions taking more parameters than the limit."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._util import parse, python_files, rel

NAME = "wide-signatures"
OWNER = ""
ROOTS = ("src/ml_stack",)
LIMIT = 8


def describe() -> str:
    return f"A signature over {LIMIT} parameters; pass a value object instead."


def _count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    args = node.args
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    return len(names) + bool(args.vararg) + bool(args.kwarg)


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
            count = _count(node)
            if count > LIMIT:
                out.append(Finding(where, node.lineno, f"{node.name}: {count} parameters"))
    return out
