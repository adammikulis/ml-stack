"""The development branch: the one the primary checkout is on. Standard library only."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

__all__ = ["DEFAULT", "development_branch"]

DEFAULT = "0.3dev"
PROTECTED = ("main", "master")


def development_branch(cwd: str | Path | None = None) -> str:
    """The branch the primary checkout is on, ``ML_STACK_DEV_BRANCH`` when set, else ``DEFAULT``.

    Never main or master: when the primary sits on one of those, the default is returned.
    """
    given = os.environ.get("ML_STACK_DEV_BRANCH", "").strip()
    if given and given not in PROTECTED:
        return given
    try:
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                cwd=cwd, capture_output=True, text=True, check=False, timeout=10).stdout.strip()
        if not common:
            return DEFAULT
        branch = subprocess.run(["git", "-C", str(Path(common).parent), "branch", "--show-current"],
                                capture_output=True, text=True, check=False, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return DEFAULT
    return branch if branch and branch not in PROTECTED else DEFAULT
