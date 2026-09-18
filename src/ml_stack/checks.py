"""A finding, and how one is shown.

`Finding` is what `ml_stack.setup` and `ml_stack.doctor` both produce. `ask()` prints
each and offers its fix, never reading a password -- a fix that needs root runs through
`sudo`, which prompts on the terminal itself.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ml_stack import home
from ml_stack.log import say

__all__ = ["CHECKOUT", "Finding", "ask", "checkout", "tilde"]

CHECKOUT = Path("~/Documents/repos/ml-stack").expanduser()
"""Where the editable install must point: the checkout, not a copy of it."""


@dataclass
class Finding:
    """One thing about this machine, and what to do if it is wrong."""

    name: str
    good: bool
    said: str
    fix: str = ""            # a shell line the reader may run
    root: bool = False       # whether that line needs sudo
    note: str = ""


def tilde(path: Path) -> str:
    """A path with the home directory written as ``~``."""
    try:
        return "~/" + str(Path(path).relative_to(home.user_home()))
    except ValueError:
        return str(path)


def checkout() -> Path:
    """The checkout this package is imported from, else the one `ml-stack-doctor` checks."""
    here = Path(__file__).resolve().parents[2]
    return here if (here / "pyproject.toml").is_file() else CHECKOUT


def ask(findings: list[Finding], *, yes: bool = False) -> int:
    """Show what was found and offer each fix, one at a time."""
    worst = 0
    for one in findings:
        mark = "ok  " if one.good else "  ! "
        say(f"{mark}{one.name}: {one.said}")
        if one.note:
            say(f"      {one.note}")
        if one.good or not one.fix:
            continue
        worst = 1
        say(f"      fix: {one.fix}")
        if not yes and not sys.stdin.isatty():
            continue
        answer = "y" if yes else input("      run it now? [y/N] ").strip().lower()
        if answer != "y":
            continue
        # sudo is run so that it prompts on this terminal. The password goes from the
        # keyboard to sudo; nothing here reads it, passes it, or keeps it.
        subprocess.run(one.fix, shell=True, check=False)
    return worst
