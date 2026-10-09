"""What the landing runner does after a crash, a failed batch or a rejected push.

The runner is the only writer of the development branch, so everything here assumes it holds the
runner lock and the branch claim: stale integration trees are removed, a request left ``running``
is settled by its SHA (landed already, or queued again), and a push that origin rejected is merged
with the new tip and re-gated before it is tried again.
"""

from __future__ import annotations

import os
from pathlib import Path

import land_check as lc
import land_finish
import land_git as lg
import land_run

from ml_stack import lock as filelock
from ml_stack.activity.gate import tree_hash

PUSH_TRIES = 3
INTEGRATION = "land/"


class RunnerLock:
    """An exclusive lock on the checkout's runner file, held by the process until released."""

    def __init__(self, root: Path) -> None:
        folder = lg.common_dir(root) / "land"
        folder.mkdir(exist_ok=True)
        self.path = folder / "runner.lock"
        self.fd: int | None = None

    def acquire(self) -> bool:
        """Take the lock; false when another runner process holds it. Idempotent for the holder."""
        if self.fd is not None:
            return True
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        if not filelock.take(fd):
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()}".encode())
        self.fd = fd
        return True

    def release(self) -> None:
        """Give the lock up."""
        if self.fd is not None:
            os.ftruncate(self.fd, 0)
            filelock.release(self.fd)
            os.close(self.fd)
            self.fd = None


def holder(root: Path) -> str:
    """Who holds the runner lock, as ``pid N``, for a message."""
    return filelock.held_by(lg.common_dir(root) / "land" / "runner.lock")


def sweep(root: Path) -> list[str]:
    """Remove every integration worktree and branch left by a batch; the names removed.

    A tree whose background full run is still alive is left for it.
    """
    state = land_finish.load(root)
    if land_finish.full_running(state):
        return []
    top = lg.primary(root).path
    gone = []
    for tree in lg.trees(root)[1:]:
        if tree.branch.startswith(INTEGRATION) or tree.path.name.endswith("-probe"):
            lg.git(top, "worktree", "remove", "--force", str(tree.path), check=False)
            gone.append(tree.branch or tree.path.name)
    lg.git(top, "worktree", "prune", check=False)
    for name in lg.lines(root, "for-each-ref", "--format=%(refname:short)", f"refs/heads/{INTEGRATION}"):
        lg.git(top, "branch", "-D", name, check=False)
        gone.append(name)
    land_run.state_file(root).unlink(missing_ok=True)
    return gone


def on_target(root: Path, sha: str, target: str) -> bool:
    """Whether the commit ``sha`` is already part of the local development branch."""
    return lg.git(root, "merge-base", "--is-ancestor", sha, target, check=False).returncode == 0


def present(root: Path, sha: str, target: str) -> bool:
    """Whether the commit ``sha`` is on the development branch, itself or as equivalent patches."""
    return lg.exists(root, sha) and (on_target(root, sha, target) or not lg.unique(root, target, sha))


def on_origin(root: Path, sha: str, target: str, remote: str) -> bool:
    """Whether the commit ``sha`` is part of the remote's copy of the development branch."""
    ref = f"{remote}/{target}"
    return lg.exists(root, ref) and lg.git(root, "merge-base", "--is-ancestor", sha, ref, check=False).returncode == 0


def failed_files(results: list[lc.Result]) -> str:
    """The failing test files of the failed checks, or their names."""
    bad = [r for r in results if r.status == "fail"]
    return ", ".join(sorted({f for r in bad for f in r.failed_files}) or [r.name for r in bad])


def regate(top: Path, base: str) -> str:
    """Run the checks the diff against ``base`` calls for; an empty string when they pass."""
    checks, _, _ = lc.build_checks(top, base, 0)
    tree = tree_hash(top)
    results = [lc.execute(top, check, tree) for check in checks]
    return failed_files(results)


def resync(root: Path, target: str, remote: str, env: dict) -> str:
    """Merge the remote's moved tip into the development branch and re-gate; why not, or an empty string.

    The merge commit is the runner's own, so a conflict or a red re-gate resets to the head before it.
    """
    top = lg.primary(root).path
    ref = f"{remote}/{target}"
    before = lg.out(top, "rev-parse", "HEAD")
    done = lg.git(top, "merge", "--no-edit", "-m", f"chore: merge {ref}", ref, check=False, env=env)
    if done.returncode:
        left = ", ".join(lg.lines(top, "diff", "--name-only", "--diff-filter=U")[:6]) or "(merge failed)"
        lg.git(top, "merge", "--abort", check=False)
        lg.git(top, "reset", "--hard", "-q", before, check=False)
        return f"{ref} moved and does not merge cleanly: {left}"
    red = regate(top, ref)
    if red:
        lg.git(top, "reset", "--hard", "-q", before, check=False)
        return f"{ref} moved; the merged tree failed its re-gate: {red}"
    return ""


def push_with_resync(runner) -> str:
    """Push the development branch; when the remote moved, merge it in, re-gate and push again.

    An empty string on success, else the reason, so the caller can hand every request to a person.
    """
    problem = runner.push()
    for _ in range(PUSH_TRIES):
        if not problem:
            return ""
        root, target, remote = runner.root, runner.target, runner.remote
        lg.git(root, "fetch", remote, target, check=False, env=runner.env)
        ref = f"{remote}/{target}"
        if not lg.exists(root, ref) or lg.git(root, "merge-base", "--is-ancestor", ref, target,
                                              check=False).returncode == 0:
            return problem
        why = resync(root, target, remote, runner.env)
        if why:
            return why
        problem = runner.push()
    return problem
