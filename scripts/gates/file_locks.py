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


def _handle_transfer_only(tree: ast.Module, binding: str) -> bool:
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    references = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Name) and node.id == binding]
    return bool(references) and all(
        isinstance(parents.get(node), ast.Attribute)
        and parents[node].value is node
        and parents[node].attr == "open_osfhandle"
        and isinstance(parents[node].ctx, ast.Load)
        for node in references)


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
                    if (alias.name.split(".")[0] in MODULES
                            and not (alias.name == "msvcrt"
                                     and _handle_transfer_only(tree, alias.asname or alias.name))):
                        out.append(Finding(where, node.lineno, f"import {alias.name}"))
            elif (isinstance(node, ast.ImportFrom) and node.module
                  and node.module.split(".")[0] in MODULES
                  and (node.module != "msvcrt" or any(
                      alias.name != "open_osfhandle" for alias in node.names))):
                out.append(Finding(where, node.lineno, f"from {node.module}"))
        for node, name in calls(tree):
            if name.endswith("flock") or dotted(node.func).endswith("flock"):
                out.append(Finding(where, node.lineno, "flock"))
            elif name == "msvcrt.locking":
                out.append(Finding(where, node.lineno, "msvcrt.locking"))
    return out
