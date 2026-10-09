"""except Exception and bare except handlers."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import dotted, parse

NAME = "broad-excepts"
OWNER = ""
INCREMENTAL = True
ROOTS = ("src/ml_stack",)


def _broad(handler: ast.ExceptHandler) -> str:
    if handler.type is None:
        return "bare except"
    kinds = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    for kind in kinds:
        name = dotted(kind)
        if name in ("Exception", "BaseException") or name.endswith(".Exception"):
            return f"except {name or 'Exception'}"
    return ""


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    tree = parse(path)
    if tree is None:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            detail = _broad(node)
            if detail:
                out.append(Finding(where, node.lineno, detail))
    return out


find = finder(ROOTS, scan)


def describe() -> str:
    return "A handler that swallows everything; catch the error the call actually raises."
