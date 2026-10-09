"""ArgumentParser constructions across the library."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import calls, dotted, parse

NAME = "argument-parsers"
OWNER = ""
INCREMENTAL = True
ROOTS = ("src/ml_stack",)


def describe() -> str:
    return "Another command-line parser; each one is a surface a caller has to learn."


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    tree = parse(path)
    if tree is None:
        return out
    for node, name in calls(tree):
        if name.endswith("ArgumentParser") or dotted(node.func).endswith("ArgumentParser"):
            out.append(Finding(where, node.lineno, "ArgumentParser()"))
    return out


find = finder(ROOTS, scan)
