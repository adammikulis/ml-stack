"""The ruff and pyright the tool-backed budgets were counted with, and how to run them."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path

PINNED = Path(__file__).resolve().parent / "pinned.txt"
INSTALL = "pip install -r scripts/gates/pinned.txt"


@cache
def pins() -> dict[str, str]:
    """Tool name -> the version pinned.txt names."""
    out: dict[str, str] = {}
    for raw in PINNED.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "==" in line:
            name, _, want = line.partition("==")
            out[name.strip()] = want.strip()
    return out


@cache
def tool(name: str) -> tuple[str, ...] | None:
    """How to run ``name`` here, or None when it is not installed."""
    found = shutil.which(name)
    if found:
        return (found,)
    probe = subprocess.run([sys.executable, "-m", name, "--version"],
                           capture_output=True, text=True, check=False)
    if probe.returncode == 0:
        return (sys.executable, "-m", name)
    return None


@cache
def version(name: str) -> str:
    """The version ``name --version`` reports, empty when it reports nothing readable."""
    cmd = tool(name)
    if cmd is None:
        return ""
    done = subprocess.run([*cmd, "--version"], capture_output=True, text=True, check=False)
    found = re.search(r"\d+\.\d+(\.\d+)?", done.stdout or done.stderr or "")
    return found.group(0) if found else ""


def skip(name: str) -> str:
    """Why ``name``'s metrics cannot be counted here, empty when they can."""
    if tool(name) is None:
        return f"{name} is not installed; {INSTALL}"
    want, have = pins().get(name, ""), version(name)
    if want and have != want:
        return (f"{name} {have or 'of an unreadable version'} is installed and the budgets "
                f"were counted with {name} {want}; {INSTALL}")
    return ""


def measured_with() -> str:
    """One line naming each pinned tool and the version answering here."""
    return ", ".join(f"{name} {version(name) or 'absent'}" for name in sorted(pins()))
