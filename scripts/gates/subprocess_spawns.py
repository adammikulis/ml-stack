"""Long-lived subprocesses started outside the modules that own launching."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._util import calls, exempt, parse, python_files, rel

NAME = "subprocess-spawns"
OWNER = "ml_stack.platform"
ROOTS = ("src/ml_stack",)
OWNS = (
    "src/ml_stack/platform.py",
    "src/ml_stack/jobs.py",
    "src/ml_stack/serve/backend.py",
)


def describe() -> str:
    return "A detached process started here; ml_stack.platform launches and records one."


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
            if name == "subprocess.Popen":
                out.append(Finding(where, node.lineno, "subprocess.Popen"))
    return out
