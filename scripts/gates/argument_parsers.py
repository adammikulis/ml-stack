"""ArgumentParser constructions across the library."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._util import calls, dotted, parse, python_files, rel

NAME = "argument-parsers"
OWNER = ""
ROOTS = ("src/ml_stack",)


def describe() -> str:
    return "Another command-line parser; each one is a surface a caller has to learn."


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        tree = parse(path)
        if tree is None:
            continue
        for node, name in calls(tree):
            if name.endswith("ArgumentParser") or dotted(node.func).endswith("ArgumentParser"):
                out.append(Finding(where, node.lineno, "ArgumentParser()"))
    return out
