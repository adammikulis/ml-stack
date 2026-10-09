"""Long-lived subprocesses started outside the modules that own launching."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import calls, exempt, parse

NAME = "subprocess-spawns"
OWNER = "ml_stack.platform"
INCREMENTAL = True
ROOTS = ("src/ml_stack",)
OWNS = (
    "src/ml_stack/platform.py",
    "src/ml_stack/jobs.py",
    "src/ml_stack/serve/backend.py",
    "src/ml_stack/serve/exit_guard.py",
    "src/ml_stack/sandbox/run.py",
)


def describe() -> str:
    return "A detached process started here; ml_stack.platform launches and records one."


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    if exempt(where, OWNS):
        return out
    tree = parse(path)
    if tree is None:
        return out
    for node, name in calls(tree):
        if name == "subprocess.Popen":
            out.append(Finding(where, node.lineno, "subprocess.Popen"))
    return out


find = finder(ROOTS, scan)
