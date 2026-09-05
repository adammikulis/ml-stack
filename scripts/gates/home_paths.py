"""Home-directory lookups outside the module that resolves paths."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._util import calls, dotted, exempt, parse, python_files, rel

NAME = "home-paths"
OWNER = "ml_stack.home"
ROOTS = ("src/ml_stack",)
OWNS = ("src/ml_stack/home.py",)


def describe() -> str:
    return "A home directory resolved in place; ml_stack.home names every directory once."


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
    return out
