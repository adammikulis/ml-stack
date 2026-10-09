"""Git queries and worktree helpers shared by the landing commands."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

PROTECTED = ("main", "master")


@dataclass(frozen=True)
class Tree:
    """A worktree: its path, the branch checked out there (empty when detached) and its head."""

    path: Path
    branch: str
    head: str


def git(root: Path | str, *args: str, check: bool = True, env: dict[str, str] | None = None,
        stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``git -C root args`` and return the completed process."""
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          check=False, env=env, input=stdin)
    if check and done.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {(done.stderr or done.stdout).strip()}")
    return done


def out(root: Path | str, *args: str) -> str:
    """Stripped standard output of a git command that must succeed."""
    return git(root, *args).stdout.strip()


def lines(root: Path | str, *args: str) -> list[str]:
    """Non-empty output lines of a git command that must succeed."""
    return [x for x in git(root, *args).stdout.splitlines() if x]


def toplevel(start: Path | str) -> Path:
    """The root of the worktree containing ``start``."""
    return Path(out(start, "rev-parse", "--show-toplevel"))


def common_dir(root: Path | str) -> Path:
    """The git directory every worktree of this repository shares."""
    return Path(out(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))


def trees(root: Path | str) -> list[Tree]:
    """Every registered worktree; the first is the primary checkout."""
    found: list[Tree] = []
    path, head, branch = "", "", ""
    for line in [*git(root, "worktree", "list", "--porcelain").stdout.splitlines(), ""]:
        if line.startswith("worktree "):
            path = line[9:]
        elif line.startswith("HEAD "):
            head = line[5:]
        elif line.startswith("branch refs/heads/"):
            branch = line[18:]
        elif not line and path:
            found.append(Tree(Path(path), branch, head))
            path, head, branch = "", "", ""
    return found


def primary(root: Path | str) -> Tree:
    """The primary checkout."""
    return trees(root)[0]


def tree_of(root: Path | str, branch: str) -> Tree | None:
    """The worktree that has ``branch`` checked out, if any."""
    return next((t for t in trees(root) if t.branch == branch), None)


def dirty(path: Path | str) -> bool:
    """Whether the worktree at ``path`` has uncommitted or untracked files."""
    return bool(git(path, "status", "--porcelain").stdout.strip())


def exists(root: Path | str, ref: str) -> bool:
    """Whether ``ref`` names a commit."""
    return git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False).returncode == 0


def refuse_protected(*names: str) -> None:
    """Raise when a branch named here is one landing never touches."""
    for name in names:
        if name in PROTECTED:
            raise ValueError(f"land never touches {name}")


def unique(root: Path | str, target: str, branch: str) -> dict[str, str]:
    """Commits on ``branch`` with no equivalent patch on ``target``, sha to patch-id."""
    plus = [x[2:] for x in lines(root, "cherry", target, branch) if x.startswith("+ ")]
    if not plus:
        return {}
    log = git(root, "log", "-p", "--no-merges", "--reverse", f"{target}..{branch}").stdout
    ids = {}
    for row in git(root, "patch-id", "--stable", stdin=log).stdout.splitlines():
        patch, _, sha = row.partition(" ")
        ids[sha] = patch
    return {sha: ids.get(sha, f"empty:{sha}") for sha in plus}


def files_of(root: Path | str, shas: list[str]) -> list[str]:
    """Sorted paths touched by the given commits."""
    touched: set[str] = set()
    for sha in shas:
        touched.update(lines(root, "show", "--no-renames", "--name-only", "--format=", sha))
    return sorted(touched)


def behind(root: Path | str, target: str, branch: str) -> int:
    """How many commits ``target`` has that ``branch`` lacks."""
    return int(out(root, "rev-list", "--count", f"{branch}..{target}"))


def merge_tree(root: Path | str, left: str, right: str) -> tuple[str, list[str]]:
    """The tree of merging ``right`` into ``left`` without a checkout and the conflicted paths."""
    done = git(root, "merge-tree", "--write-tree", "--name-only", "--no-messages", left, right,
               check=False)
    rows = done.stdout.splitlines()
    if done.returncode not in (0, 1) or not rows:
        raise RuntimeError(f"git merge-tree {left} {right}: {done.stderr.strip()}")
    return rows[0], rows[1:] if done.returncode else []


def synthetic_merge(root: Path | str, tree: str, left: str, right: str) -> str:
    """A parentless-of-refs merge commit holding ``tree``, used to chain predicted merges."""
    return out(root, "commit-tree", tree, "-p", left, "-p", right, "-m", "plan")
