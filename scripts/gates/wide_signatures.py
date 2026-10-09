"""Functions taking more parameters than the limit."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import parse

NAME = "wide-signatures"
OWNER = ""
INCREMENTAL = True
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


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    tree = parse(path)
    if tree is None:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        count = _count(node)
        if count > LIMIT:
            out.append(Finding(where, node.lineno, f"{node.name}: {count} parameters"))
    return out


find = finder(ROOTS, scan)
