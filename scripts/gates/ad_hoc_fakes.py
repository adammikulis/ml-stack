"""Fakes defined in the suite instead of the shared fakes module."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._util import parse, python_files, rel

NAME = "ad-hoc-fakes"
OWNER = "ml_stack.testing.fakes"
ROOTS = ("tests",)


def describe() -> str:
    return "A fake written in the suite; ml_stack.testing.fakes holds the shared ones."


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        tree = parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith("Fake"):
                out.append(Finding(where, node.lineno, f"class {node.name}"))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("fake_"):
                    out.append(Finding(where, node.lineno, f"def {node.name}"))
    return out
