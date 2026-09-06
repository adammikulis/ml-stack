"""Pyright's errors over src/ml_stack, under the checks configured in pyproject.toml."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import Finding, _pins
from ._util import rel

NAME = "pyright-errors"
OWNER = ""


def command() -> tuple[str, ...] | None:
    """How to run pyright here, or None when it is not installed."""
    return _pins.tool("pyright")


def skip() -> str:
    """Why pyright cannot be counted here, empty when it can."""
    return _pins.skip("pyright")


def describe() -> str:
    return "A pyright error: an undefined name, an unbound variable, an __all__ that names nothing."


def find(root: Path) -> list[Finding]:
    cmd = command()
    if cmd is None or not (root / "src" / "ml_stack").is_dir():
        return []
    done = subprocess.run([*cmd, "--outputjson"], cwd=root,
                          capture_output=True, text=True, check=False)
    if not done.stdout.strip():
        raise RuntimeError(f"pyright exited {done.returncode}: {done.stderr.strip()}")
    here = root.resolve()
    out = []
    for item in json.loads(done.stdout).get("generalDiagnostics", []):
        if item.get("severity") != "error":
            continue
        line = ((item.get("range") or {}).get("start") or {}).get("line", 0) + 1
        rule = item.get("rule") or "error"
        out.append(Finding(rel(Path(item["file"]).resolve(), here), line,
                           f"{rule} {item['message'].splitlines()[0]}"))
    return out
