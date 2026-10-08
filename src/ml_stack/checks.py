"""A finding, and how one is shown.

`Finding` is what `ml_stack.setup` and `ml_stack.doctor` both produce. `ask()` prints
each and offers its fix, never reading a password -- a fix that needs root runs through
`sudo`, which prompts on the terminal itself.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ml_stack import home
from ml_stack.log import say

__all__ = ["Finding", "ask", "checkout", "line", "tilde"]


@dataclass
class Finding:
    """One thing about this machine, and what to do if it is wrong."""

    name: str
    good: bool
    said: str
    fix: list[str] = field(default_factory=list)  # the command that repairs it, as an argument vector
    cwd: str = ""            # the directory that command runs in
    root: bool = False       # whether that command needs sudo
    note: str = ""


def tilde(path: Path) -> str:
    """A path with the home directory written as ``~``."""
    try:
        return "~/" + str(Path(path).relative_to(home.user_home()))
    except ValueError:
        return str(path)


def line(fix: list[str], cwd: str = "") -> str:
    """The command a fix runs, written as a shell line for display."""
    text = shlex.join(fix)
    return f"cd {shlex.quote(cwd)} && {text}" if cwd else text


def checkout() -> Path | None:
    """The checkout this package is imported from, or None for an installed copy."""
    here = Path(__file__).resolve().parents[2]
    return here if (here / "pyproject.toml").is_file() else None


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
        say(f"      fix: {line(one.fix, one.cwd)}")
        if not yes and not sys.stdin.isatty():
            continue
        answer = "y" if yes else input("      run it now? [y/N] ").strip().lower()
        if answer != "y":
            continue
        # a sudo in the command prompts on this terminal; nothing here reads the password
        subprocess.run(one.fix, cwd=one.cwd or None, check=False)
    return worst
