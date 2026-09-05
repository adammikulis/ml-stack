"""Process-table scans outside the module that owns them."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._util import calls, dotted, exempt, parse, python_files, rel

NAME = "process-scans"
OWNER = "ml_stack.serve.process"
ROOTS = ("src/ml_stack",)
OWNS = ("src/ml_stack/serve/process.py",)


def describe() -> str:
    return "A walk of the process table; ml_stack.serve.process finds and stops servers."


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
            if name == "psutil.process_iter" or dotted(node.func).endswith("psutil.process_iter"):
                out.append(Finding(where, node.lineno, "psutil.process_iter"))
    return out
