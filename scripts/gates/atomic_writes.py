"""Hand-rolled atomic file replacement outside the file helpers."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._util import calls, exempt, parse, python_files, rel

NAME = "atomic-writes"
OWNER = "ml_stack.files.write_json"
ROOTS = ("src/ml_stack",)
OWNS = ("src/ml_stack/files.py",)
TARGETS = {"os.replace", "os.rename"}


def describe() -> str:
    return "A write-then-replace done by hand; ml_stack.files.write_json is the atomic write."


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
            if name in TARGETS:
                out.append(Finding(where, node.lineno, name))
    return out
