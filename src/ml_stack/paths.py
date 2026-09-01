"""Where a path sits, for the files that must not land in the wrong place.

An export that holds real names -- a bench run, an alias file, anything read out of a
live community -- must never be written where a commit could pick it up. The question
every such writer asks is the same one: is this destination inside a repository?
"""

from __future__ import annotations

from pathlib import Path


def repo_root(path: str | Path) -> Path | None:
    """The git working tree `path` sits in -- the nearest ancestor holding `.git` -- or None.

    `.git` may be a directory or a file (a worktree or submodule keeps a file pointing at
    the real one), and both count. The path is resolved first, so a symlink into a
    repository is inside it. A path that does not exist yet is judged by its nearest
    existing ancestor, which is what matters for a file about to be written. Named in a
    refusal so the person can see which repository they were about to write into.
    """
    where = Path(path).expanduser().resolve()
    while not where.exists() and where.parent != where:
        where = where.parent
    for parent in (where, *where.parents):
        if (parent / ".git").exists():
            return parent
    return None


def inside_a_repo(path: str | Path) -> bool:
    """Whether `path` sits inside a git working tree.

    For a destination that must not be committed: refuse it when this is true. The
    reasoning is `repo_root`'s; this is the yes-or-no most callers want.
    """
    return repo_root(path) is not None
