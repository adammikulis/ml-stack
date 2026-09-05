"""File locking outside the lock module."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._util import calls, dotted, exempt, parse, python_files, rel

NAME = "file-locks"
OWNER = "ml_stack.lock"
ROOTS = ("src/ml_stack",)
OWNS = ("src/ml_stack/lock.py",)
MODULES = {"fcntl", "msvcrt"}


def describe() -> str:
    return "A lock taken on a file by hand; ml_stack.lock holds one across both platforms."


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        if exempt(where, OWNS):
            continue
        tree = parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in MODULES:
                        out.append(Finding(where, node.lineno, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in MODULES:
                    out.append(Finding(where, node.lineno, f"from {node.module}"))
        for node, name in calls(tree):
            if name.endswith("flock") or dotted(node.func).endswith("flock"):
                out.append(Finding(where, node.lineno, "flock"))
    return out
