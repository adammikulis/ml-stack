"""The counts that may not fall, where every budgeted metric may not rise."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

from ._util import dotted, parse, python_files

NAME = "tests-collected"
OWNER = ""
ROOTS = ("tests",)
COUNTED = re.compile(r"(\d+)(?:/(\d+))? tests? collected")
ERRORED = re.compile(r"(\d+) errors?\b")


def describe() -> str:
    return ("Tests pytest collects, deselected ones included. A floor: it may not fall. "
            "A module that stops importing takes its tests out of the suite silently.")


def guards(root: Path) -> list[str]:
    """Every import a test module waits on before it will collect at all."""
    out = set()
    for path in python_files(root, ROOTS):
        tree = parse(path)
        if tree is None:
            continue
        for node in tree.body:
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call) or dotted(inner.func) != "pytest.importorskip":
                    continue
                first = inner.args[0] if inner.args else None
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    out.add(first.value)
    return sorted(out)


def installed(name: str) -> bool:
    """Whether this module can be imported here."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def skip(root: Path) -> str:
    """Why the collected count is not comparable here, empty when it is."""
    if not (root / "tests").is_dir():
        return "there is no tests/ here to collect"
    missing = [name for name in guards(root) if not installed(name)]
    if not missing:
        return ""
    return (f"{', '.join(missing)} absent, so those modules collect nothing here; the "
            "count is a floor only where every extra the suite reads is installed")


def collect(root: Path) -> tuple[int, int]:
    """How many tests pytest collects under root, and how many modules failed to."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(root / "src"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    done = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                           "-p", "no:cacheprovider"],
                          cwd=root, capture_output=True, text=True, check=False, env=env)
    found = COUNTED.search(done.stdout)
    if not found:
        raise RuntimeError(f"pytest --collect-only exited {done.returncode}: "
                           f"{(done.stdout + done.stderr).strip()[-800:]}")
    broke = ERRORED.search(done.stdout)
    return int(found.group(2) or found.group(1)), int(broke.group(1)) if broke else 0
