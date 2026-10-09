"""Home-directory lookups and state-root path literals outside the module that resolves paths."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import calls, dotted, exempt, parse

NAME = "home-paths"
OWNER = "poolhouse.home"
INCREMENTAL = True
ROOTS = ("src/poolhouse",)
OWNS = ("src/poolhouse/home.py",)


def describe() -> str:
    return ("A home directory resolved in place, or the state root written as a literal path; "
            "poolhouse.home names every directory once.")


def state_literal(value: object) -> bool:
    """Whether a string constant names the state root as a path rather than asking for it."""
    if not isinstance(value, str):
        return False
    return value in ("~/.poolhouse", ".poolhouse") or value.startswith(("~/.poolhouse/", ".poolhouse/"))


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    if exempt(where, OWNS):
        return out
    tree = parse(path)
    if tree is None:
        return out
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


find = finder(ROOTS, scan)
