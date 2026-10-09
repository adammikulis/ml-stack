"""Long-lived subprocesses started outside the modules that own launching."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import calls, exempt, parse

NAME = "subprocess-spawns"
OWNER = "poolhouse.platform"
INCREMENTAL = True
ROOTS = ("src/poolhouse",)
OWNS = (
    "src/poolhouse/platform.py",
    "src/poolhouse/jobs.py",
    "src/poolhouse/serve/backend.py",
    "src/poolhouse/serve/exit_guard.py",
    "src/poolhouse/sandbox/run.py",
)


def describe() -> str:
    return "A detached process started here; poolhouse.platform launches and records one."


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
