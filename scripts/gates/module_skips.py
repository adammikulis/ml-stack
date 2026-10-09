"""Skips outside a test function, which take a whole module out of collection."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import dotted, parse

NAME = "module-skips"
OWNER = ""
INCREMENTAL = True
ROOTS = ("tests",)

SKIPS = ("pytest.importorskip", "pytest.skip", "pytest.mark.skip", "pytest.mark.skipif",
         "pytest.mark.xfail")


def describe() -> str:
    return ("A skip at module level; the file collects no tests at all and the count of "
            "what ran falls silently. Guard the tests that need the import, one by one.")


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    tree = parse(path)
    if tree is None:
        return out
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and dotted(inner.func) in SKIPS:
                out.append(Finding(where, inner.lineno, f"{dotted(inner.func)}()"))
    return out


find = finder(ROOTS, scan)
