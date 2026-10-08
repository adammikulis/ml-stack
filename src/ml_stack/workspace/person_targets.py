"""The exact target of an authorizable action, derived from the checkout and never from anything a model wrote."""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["development_branch", "project_of", "push_dev_target"]


def _git(cwd: str | Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False,
                              timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def project_of(cwd: str | Path) -> str:
    """The primary checkout of the repository ``cwd`` is in, or ``cwd`` when it is in none."""
    common = _git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return str(Path(common).parent) if common else str(cwd)


def development_branch(cwd: str | Path) -> str:
    """The branch the primary checkout is on; empty when it is detached or ``main``."""
    branch = _git(project_of(cwd), "branch", "--show-current")
    return "" if branch == "main" else branch


def push_dev_target(cwd: str | Path, remote: str = "") -> str:
    """``remote:branch`` for pushing the development branch; empty when there is none to push."""
    branch = development_branch(cwd)
    if not branch:
        return ""
    return f"{remote or _git(project_of(cwd), 'config', f'branch.{branch}.remote') or 'origin'}:{branch}"
