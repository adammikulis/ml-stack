"""The facts a guard derives from the checkout it runs in, never from anything a model wrote."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Checkout", "git_ok", "inspect", "run_git"]

GIT_TIMEOUT_S = 2.0
HOOK_VARIABLES = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")


@dataclass(frozen=True, slots=True)
class Checkout:
    """The primary checkout of a repository and the remote its ``main`` pushes to."""

    project: str
    remote: str


def _run(cwd: str | Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        env = {k: v for k, v in os.environ.items() if k not in HOOK_VARIABLES}
        return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False,
                              timeout=GIT_TIMEOUT_S, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None


def run_git(cwd: str | Path, *args: str) -> str:
    """The stdout of a bounded git call; empty when it fails."""
    done = _run(cwd, *args)
    return done.stdout.strip() if done is not None and done.returncode == 0 else ""


def git_ok(cwd: str | Path, *args: str) -> bool:
    """Whether a bounded git call exits 0."""
    done = _run(cwd, *args)
    return done is not None and done.returncode == 0


def inspect(cwd: str | Path) -> Checkout:
    """The checkout ``cwd`` belongs to, from two bounded git calls."""
    common = run_git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir")
    project = str(Path(common).parent) if common else str(cwd)
    remote = run_git(project, "config", "branch.main.remote") if common else ""
    return Checkout(project, remote or "origin")
