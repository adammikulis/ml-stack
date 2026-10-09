"""The worktree registry as the hooks use it: claim a tree, finish an owner's trees, show orphans and deliver notices.

Run by SessionStart, SubagentStart, SubagentStop, the lead's attention hook and the git post-commit
hook. Every call is bounded and best effort: a failure is a warning, never a refused tool call.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import workspace_hook

from ml_stack.workspace import (
    integration_git as repo,
    trees,
    trees_notice,
)

FAILURES = (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired)
AGENT = "ML_STACK_WORKSPACE_AGENT"


def top(cwd: str) -> Path | None:
    """The root of the checkout holding ``cwd``, or None when it is not a git checkout."""
    try:
        return Path(repo.git(Path(cwd or "."), "rev-parse", "--show-toplevel"))
    except FAILURES:
        return None


def deliver(found: list[trees_notice.Notice], stage: str, environment: dict | None) -> None:
    """Post each notice to the board and message each named recipient."""
    for note in found:
        workspace_hook.run(["ml-stack-workspace", "announce", "milestone", note.text[:200]], stage,
                           environment=environment)
        for name in note.to:
            workspace_hook.run(["ml-stack-workspace", "dm", name, note.text], stage, environment=environment)


def check(stage: str, cwd: str, environment: dict | None = None, *, here: bool = False, lead: str = "") -> str:
    """Scan, record the lead, deliver the notices due and return the orphan lines for a status page."""
    root = top(cwd)
    if root is None:
        return ""
    owner = (environment or os.environ).get(AGENT, "")
    try:
        if lead:
            trees.set_lead(root, lead)
        claimed = (owner, root) if here and owner else None
        deliver(trees_notice.notify(root, time.time(), claimed, here), stage, environment)
        lines = [] if here else trees.lines(root, time.time())
    except FAILURES as error:
        workspace_hook.warning(stage, error)
        return ""
    return "Orphan worktrees (owner stopped, work undecided; data):\n" + "\n".join(lines) if lines else ""


def started(stage: str, cwd: str, name: str, purpose: str) -> None:
    """Register the tree a subagent was started in as owned by it; a primary checkout registers nothing."""
    root = top(cwd)
    if root is None or not name:
        return
    try:
        trees.claim(root, root, name, time.time(), {"purpose": purpose})
        trees.scan(root, time.time())
    except FAILURES as error:
        workspace_hook.warning(stage, error)


def stopped(stage: str, cwd: str, environment: dict) -> None:
    """Mark the stopping subagent's trees finished; unlanded or dirty work goes to the board and the lead now."""
    root, name = top(cwd), environment.get(AGENT, "")
    if root is None or not name:
        return
    try:
        debt = trees.finish(root, name, time.time())
        for row in debt:
            text = trees.line(row, time.time())
            workspace_hook.run(["ml-stack-workspace", "announce", "milestone", text[:200]], stage,
                               environment=environment)
            lead = trees_notice.lead_of(trees.recorded(root))
            if lead:
                workspace_hook.run(["ml-stack-workspace", "dm", lead, text], stage, environment=environment)
        deliver(trees_notice.notify(root, time.time()), stage, environment)
    except FAILURES as error:
        workspace_hook.warning(stage, error)


def main(argv: list[str]) -> int:
    """The git post-commit entry: check only the tree that was committed in."""
    if argv[:1] == ["post-commit"]:
        check("post-commit", str(Path.cwd()), None, here=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
