"""Skips outside a test function, which take a whole module out of collection."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._util import dotted, parse, python_files, rel

NAME = "module-skips"
OWNER = ""
ROOTS = ("tests",)

SKIPS = ("pytest.importorskip", "pytest.skip", "pytest.mark.skip", "pytest.mark.skipif",
         "pytest.mark.xfail")


def describe() -> str:
    return ("A skip at module level; the file collects no tests at all and the count of "
            "what ran falls silently. Guard the tests that need the import, one by one.")


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        tree = parse(path)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and dotted(inner.func) in SKIPS:
                    out.append(Finding(where, inner.lineno, f"{dotted(inner.func)}()"))
    return out
