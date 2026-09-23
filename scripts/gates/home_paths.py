"""Home-directory lookups and state-root path literals outside the module that resolves paths."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._util import calls, dotted, exempt, parse, python_files, rel

NAME = "home-paths"
OWNER = "ml_stack.home"
ROOTS = ("src/ml_stack",)
OWNS = ("src/ml_stack/home.py",)


def describe() -> str:
    return ("A home directory resolved in place, or the state root written as a literal path; "
            "ml_stack.home names every directory once.")


def state_literal(value: object) -> bool:
    """Whether a string constant names the state root as a path rather than asking for it."""
    if not isinstance(value, str):
        return False
    return value in ("~/.ml-stack", ".ml-stack") or value.startswith(("~/.ml-stack/", ".ml-stack/"))


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        if exempt(where, OWNS):
            continue
        tree = parse(path)
        if tree is None:
            continue
        for node, name in calls(tree):
            plain = dotted(node.func)
            if name.endswith("Path.home") or plain.endswith("Path.home"):
                out.append(Finding(where, node.lineno, "Path.home()"))
            elif plain == "expanduser" or plain.endswith(".expanduser"):
                out.append(Finding(where, node.lineno, "expanduser()"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and state_literal(node.value):
                out.append(Finding(where, node.lineno, f"{node.value!r}"))
    return out
