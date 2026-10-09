"""Worktrees whose owner stopped, that hold work nobody landed, bundled or abandoned."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from . import Finding

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from poolhouse import trees

NAME = "orphan-trees"
OWNER = "scripts/worktrees"
INCREMENTAL = True
HARD = True


def describe() -> str:
    return ("A worktree whose owner stopped more than the grace (2 hours by default) ago and which is not "
            "landed, bundled or abandoned. Decide it: scripts/worktrees close TREE --landed, --bundle or "
            "--abandon 'reason'. There is no allowance.")


def find(root: Path) -> list[Finding]:
    now = time.time()
    try:
        found = trees.rows(root, now)
        late = trees.past_grace(found, trees.current_policy(root), now)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError):
        return []
    return [Finding(f"worktrees/{Path(r['path']).name}", 1, trees.line(r, now) + " -- run: " + trees.suggested(r)) for r in late]
