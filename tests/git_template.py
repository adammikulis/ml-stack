"""A built git checkout, copied per test instead of rebuilt.

``git init``, an add, a commit and a ``worktree add`` are four processes a test paid for every
time it asked for a baseline checkout (about 120 ms each on a quiet machine, several times that
under load). The first caller in a process builds the checkout once in a temporary directory; every
caller after copies the tree (a few milliseconds) and ``git worktree repair`` re-points the
linked worktrees' absolute paths at the copy.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from ml_stack.net import git

_BUILT: dict[str, Path] = {}


def _built(key: str, build: Callable[[Path], None]) -> Path:
    if key not in _BUILT:
        where = Path(tempfile.mkdtemp(prefix=f"git-template-{key}-"))
        atexit.register(shutil.rmtree, where, True)
        build(where)
        _BUILT[key] = where
    return _BUILT[key]


def copy_checkout(key: str, build: Callable[[Path], None], dest: Path, primary: str,
                  *beside: str) -> None:
    """Make ``dest`` hold what ``build`` makes. ``build(root)`` creates the checkout ``root/primary``
    and any directory ``root/<beside>`` (a linked worktree kept beside it); a linked worktree inside
    the primary comes with it. Every linked worktree the template lists is repaired afterwards so
    its absolute paths point at the copy, not at the template."""
    template = _built(key, build)
    dest.mkdir(parents=True, exist_ok=True)
    for name in (primary, *beside):
        shutil.copytree(template / name, dest / name, symlinks=True)
    listed = git.run(["worktree", "list", "--porcelain"], cwd=template / primary).stdout
    linked = [line[len("worktree "):] for line in listed.splitlines() if line.startswith("worktree ")][1:]
    moved = [str(dest / Path(path).resolve().relative_to(template.resolve())) for path in linked]
    if moved:
        git.run(["worktree", "repair", *moved], cwd=dest / primary)
