"""Whether a change reaches the gate at all.

The gate checks budgets, the red-team map, the command reference and the layer rules, all of which
read source, tests and generated files. A change to prose no test names (HANDOFF.md, a guide) cannot
move any of them, and `scripts/affected.py` already knows which paths those are, so the gate says so
and stops instead of spending a minute re-proving the tree. Anything it cannot place runs the gate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import affected


def base_ref(root: Path, base: str) -> str:
    """The remote tip of ``base`` when there is one, since the local branch may lag it."""
    remote = f"origin/{base}"
    found = subprocess.run(["git", "rev-parse", "--verify", "--quiet", remote], cwd=root,
                           capture_output=True, check=False)
    return remote if found.returncode == 0 else base


def unreached(root: Path, base: str) -> str:
    """Why the gate has nothing to check for this diff, or '' when it must run."""
    try:
        changed, deleted = affected.changed_since(root, base_ref(root, base))
    except (RuntimeError, IndexError, OSError):
        return ""
    if not changed:
        return ""
    picked = affected.select(root, changed, deleted, base=base_ref(root, base))
    if picked.files or picked.unmapped:
        return ""
    shown = ", ".join(changed[:5]) + (f" and {len(changed) - 5} more" if len(changed) > 5 else "")
    return f"the change reaches no test and no generated file ({shown})"
