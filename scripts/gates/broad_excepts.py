"""except Exception and bare except handlers."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._util import dotted, parse, python_files, rel

NAME = "broad-excepts"
OWNER = ""
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


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        tree = parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                detail = _broad(node)
                if detail:
                    out.append(Finding(where, node.lineno, detail))
    return out


def describe() -> str:
    return "A handler that swallows everything; catch the error the call actually raises."
