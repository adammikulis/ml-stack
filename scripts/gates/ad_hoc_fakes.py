"""Fakes defined in the suite instead of the shared fakes module."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import parse

NAME = "ad-hoc-fakes"
OWNER = "ml_stack.testing.fakes"
INCREMENTAL = True
ROOTS = ("tests",)


def describe() -> str:
    return "A fake written in the suite; ml_stack.testing.fakes holds the shared ones."


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    tree = parse(path)
    if tree is None:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith("Fake"):
            out.append(Finding(where, node.lineno, f"class {node.name}"))
        elif (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
              and node.name.startswith("fake_")):
            out.append(Finding(where, node.lineno, f"def {node.name}"))
    return out


find = finder(ROOTS, scan)
