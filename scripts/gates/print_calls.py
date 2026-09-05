"""print() in library code, outside the modules a console script runs."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._util import calls, exempt, parse, python_files, rel

NAME = "print-calls"
OWNER = ""
ROOTS = ("src/ml_stack",)

COMMANDS = {
    "src/ml_stack/claude.py",
    "src/ml_stack/cli.py",
    "src/ml_stack/do.py",
    "src/ml_stack/doctor.py",
    "src/ml_stack/fleet/app.py",
    "src/ml_stack/fleet/daemon.py",
    "src/ml_stack/fleet/join.py",
    "src/ml_stack/fleet/peers.py",
    "src/ml_stack/graph/bench/run.py",
    "src/ml_stack/graph/serve.py",
    "src/ml_stack/graph/store_cli.py",
    "src/ml_stack/harness.py",
    "src/ml_stack/hub.py",
    "src/ml_stack/ingest/__init__.py",
    "src/ml_stack/jobs.py",
    "src/ml_stack/mcp.py",
    "src/ml_stack/redact/audit.py",
    "src/ml_stack/serve/cli.py",
    "src/ml_stack/setup.py",
    "src/ml_stack/suite.py",
    "src/ml_stack/train/run.py",
    "src/ml_stack/train/tools.py",
    "src/ml_stack/world/cli.py",
}


def describe() -> str:
    return "A print() in library code; return the text or log it, a caller cannot silence this."


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        if where in COMMANDS or exempt(where, tuple(COMMANDS)):
            continue
        tree = parse(path)
        if tree is None:
            continue
        for node, name in calls(tree):
            if name == "print":
                out.append(Finding(where, node.lineno, "print()"))
    return out
