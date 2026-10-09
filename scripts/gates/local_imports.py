"""poolhouse imports inside a function body."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import parse

NAME = "local-imports"
OWNER = ""
INCREMENTAL = True
ROOTS = ("src/poolhouse",)


def describe() -> str:
    return ("A poolhouse import inside a function, or an import_module of a literal poolhouse "
            "name; a deferred import hides an import cycle.")


_LOADERS = {"import_module", "__import__"}


def _loaded(node: ast.Call) -> list[str]:
    """The poolhouse module an ``import_module``/``__import__`` call names by a literal."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name not in _LOADERS or not node.args:
        return []
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str) \
            and first.value.split(".")[0] == "poolhouse":
        return [first.value]
    return []


def _modules(node: ast.AST) -> list[str]:
    """The poolhouse modules one import statement or loader call defers, else empty."""
    if isinstance(node, ast.Call):
        return _loaded(node)
    if isinstance(node, ast.Import):
        return [a.name for a in node.names if a.name.split(".")[0] == "poolhouse"]
    if isinstance(node, ast.ImportFrom):
        if node.level:
            return ["." * node.level + (node.module or "")]
        if (node.module or "").split(".")[0] == "poolhouse":
            return [node.module or ""]
    return []


def _imports(body: list[ast.stmt]) -> list[tuple[str, ast.stmt]]:
    """Each poolhouse module this body defers, once, with the statement that named it.

    A module named twice in one function is one deferred dependency however many statements
    say so, so splitting ``from x import a, b as c`` in two does not change the count.
    """
    found: list[tuple[str, ast.stmt]] = []
    seen: set[str] = set()
    for statement in body:
        for node in ast.walk(statement):
            for module in _modules(node):
                if module in seen:
                    continue
                seen.add(module)
                found.append((module, node))
    return found


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    tree = parse(path)
    if tree is None:
        return out
    seen: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for module, found in _imports(node.body):
            if (node.name, module) in seen:
                continue
            seen.add((node.name, module))
            out.append(Finding(where, found.lineno, f"{module} in {node.name}()"))
    return out


find = finder(ROOTS, scan)
