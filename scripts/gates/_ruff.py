"""Ruff's report over the tree, read once and split by linter prefix."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path

from . import Finding
from ._util import rel

ROOTS = ("src", "tests", "scripts", "packaging")


@cache
def command() -> tuple[str, ...] | None:
    """How to run ruff here, or None when it is not installed."""
    found = shutil.which("ruff")
    if found:
        return (found,)
    probe = subprocess.run([sys.executable, "-m", "ruff", "--version"],
                           capture_output=True, text=True, check=False)
    if probe.returncode == 0:
        return (sys.executable, "-m", "ruff")
    return None


def skip() -> str:
    """Why the ruff metrics cannot be counted here, empty when they can."""
    if command() is None:
        return "ruff is not installed; pip install ruff"
    return ""


@cache
def report(root: Path) -> tuple[tuple[str, int, str, str], ...]:
    """Every violation as (path, line, code, message)."""
    cmd = command()
    if cmd is None:
        return ()
    targets = [name for name in ROOTS if (root / name).exists()]
    done = subprocess.run([*cmd, "check", "--no-cache", "--output-format", "json", *targets],
                          cwd=root, capture_output=True, text=True, check=False)
    if done.returncode not in (0, 1):
        raise RuntimeError(f"ruff exited {done.returncode}: {done.stderr.strip()}")
    here = root.resolve()
    out = []
    for item in json.loads(done.stdout):
        line = (item.get("location") or {}).get("row") or 0
        out.append((rel(Path(item["filename"]).resolve(), here), line,
                    item["code"] or "", item["message"]))
    return tuple(out)


def linter(code: str) -> str:
    """The letters in front of a rule code: BLE001 -> BLE, PLR0913 -> PLR."""
    return code.rstrip("0123456789")


def findings(root: Path, prefixes: frozenset[str], invert: bool = False) -> list[Finding]:
    """Every violation whose linter prefix is (or, inverted, is not) one of these."""
    return [
        Finding(path, line, f"{code} {message}")
        for path, line, code, message in report(root)
        if (linter(code) in prefixes) is not invert
    ]
