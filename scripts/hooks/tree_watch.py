"""The worktree registry as the hooks use it: claim a tree, finish an owner's trees, show trees needing a decision and deliver notices.

Run by SessionStart, SubagentStart, SubagentStop, the lead's attention hook and the git post-commit
hook. Every call is bounded by an alarm and fails open: a locked, missing or garbled registry, a slow
``git worktree list`` or a board that cannot be reached is one warning line, never a refused tool
call, commit or push.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import workspace_hook

from poolhouse import trees, trees_notice

FAILURES = (RuntimeError, OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.TimeoutExpired)
AGENT = "POOLHOUSE_WORKSPACE_AGENT"
BUDGET_S = 8
DETACH = True
"""Whether delivery runs in a detached process: the board commands take seconds and must not hold up a hook."""


@contextlib.contextmanager
def bounded(seconds: int = BUDGET_S) -> Iterator[None]:
    """Raise TimeoutError in the body after ``seconds`` (where the platform has SIGALRM)."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def stop(_number: int, _frame: object) -> None:
        raise TimeoutError("the worktree registry did not answer in time")

    old = signal.signal(signal.SIGALRM, stop)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def top(cwd: str) -> Path | None:
    """The root of the checkout holding ``cwd``, or None when it is not a git checkout."""
    try:
        return Path(trees.git(Path(cwd or "."), "rev-parse", "--show-toplevel"))
    except FAILURES:
        return None


def deliver(found: list[trees_notice.Notice], stage: str, environment: dict | None) -> None:
    """Post each notice to the board (when it earned a board slot) and message each named recipient."""
    for note in found:
        if note.board:
            workspace_hook.run(["poolhouse-workspace", "announce", "milestone", note.text[:200]], stage,
                               environment=environment)
        for name in note.to:
            workspace_hook.run(["poolhouse-workspace", "dm", name, note.text], stage, environment=environment)


def send(found: list[trees_notice.Notice], stage: str, environment: dict | None) -> None:
    """Deliver the notices without waiting for the board: a detached process runs ``deliver``."""
    if not found:
        return
    if not DETACH:
        deliver(found, stage, environment)
        return
    payload = json.dumps([dataclasses.asdict(n) for n in found])
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "deliver", stage, payload],
                     env=environment or os.environ.copy(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)


def gather(root: Path, owner: str, here: bool, lead: str = "") -> list[trees_notice.Notice]:
    """The notices due now, for every tree or only ``root``'s; records the lead when given."""
    if lead:
        trees.set_lead(root, lead)
    return trees_notice.notify(root, time.time(), (owner, root) if here else None, here)


def check(stage: str, cwd: str, environment: dict | None = None, *, here: bool = False, lead: str = "") -> str:
    """Scan, record the lead, deliver the notices due and return the lines for trees needing a decision."""
    root = top(cwd)
    if root is None:
        return ""
    owner = (environment or os.environ).get(AGENT, "")
    try:
        with bounded():
            send(gather(root, owner, here, lead), stage, environment)
            lines = [] if here else trees.lines(root, time.time())
    except FAILURES as error:
        workspace_hook.warning(stage, error)
        return ""
    return ("Worktrees needing a decision (finished and waiting to land, or orphaned; data):\n"
            + "\n".join(lines)) if lines else ""


def started(stage: str, cwd: str, name: str, purpose: str) -> None:
    """Register the tree a subagent was started in as owned by it; a primary checkout registers nothing."""
    root = top(cwd)
    if root is None or not name:
        return
    try:
        with bounded():
            trees.claim(root, root, name, time.time(), {"purpose": purpose})
            trees.scan(root, time.time())
    except FAILURES as error:
        workspace_hook.warning(stage, error)


def stopped(stage: str, cwd: str, environment: dict) -> None:
    """Mark the stopping subagent's trees finished; unlanded or dirty work goes to the board and the lead as
    "finished, waiting to land", through the same capped notices as everything else."""
    root, name = top(cwd), environment.get(AGENT, "")
    if root is None or not name:
        return
    try:
        with bounded():
            trees.finish(root, name, time.time())
            send(trees_notice.notify(root, time.time()), stage, environment)
    except FAILURES as error:
        workspace_hook.warning(stage, error)


def main(argv: list[str]) -> int:
    """The git post-commit entry: check only the tree that was committed in.

    A person committing with plain git has no agent identity: a crossed threshold is one line on
    stderr and nothing is posted. Nothing is printed when no threshold is crossed.
    """
    if argv[:1] == ["deliver"]:
        deliver([trees_notice.Notice(**d) for d in json.loads(argv[2])], argv[1], None)
        return 0
    if argv[:1] != ["post-commit"]:
        return 0
    owner = os.environ.get(AGENT, "")
    if owner:
        check("post-commit", str(Path.cwd()), None, here=True)
        return 0
    root = top(str(Path.cwd()))
    try:
        with bounded():
            for note in gather(root, "", True) if root else []:
                print(f"worktrees: {note.text}", file=sys.stderr)
    except FAILURES as error:
        workspace_hook.warning("post-commit", error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
