"""Process-table scans outside the module that owns them."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import calls, dotted, exempt, parse

NAME = "process-scans"
OWNER = "ml_stack.serve.process"
INCREMENTAL = True
ROOTS = ("src/ml_stack",)
OWNS = ("src/ml_stack/serve/process.py",)


def describe() -> str:
    return "A walk of the process table; ml_stack.serve.process finds and stops servers."


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    if exempt(where, OWNS):
        return out
    tree = parse(path)
    if tree is None:
        return out
    for node, name in calls(tree):
        if name == "psutil.process_iter" or dotted(node.func).endswith("psutil.process_iter"):
            out.append(Finding(where, node.lineno, "psutil.process_iter"))
    return out


find = finder(ROOTS, scan)
