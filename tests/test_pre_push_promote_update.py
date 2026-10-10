"""An agent moves an existing promote/* snapshot forward; a rewrite of it, `main` and a work branch stay refused.

The change is to a protected hook, so it ships as docs/hooks-protected.patch for the owner to apply. These
tests apply it to a copy of the hooks (or use the hooks as they are once it is applied) and push through that copy.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PATCH = REPO / "docs" / "hooks-protected.patch"
ZERO = "0" * 40


def git(where: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(where), *args], check=True, text=True,
                          capture_output=True, env=env).stdout.strip()


def commit(where: Path, name: str) -> str:
    (where / name).write_text(name)
    git(where, "add", name)
    git(where, "commit", "-q", "-m", name)
    return git(where, "rev-parse", "HEAD")


@pytest.fixture
def hooks(tmp_path) -> Path:
    """A copy of scripts/hooks with the patch applied unless the real hooks already carry it."""
    root = tmp_path / "copy"
    shutil.copytree(REPO / "scripts" / "hooks", root / "scripts" / "hooks")
    patch = ["git", "apply", "-p1", str(PATCH)]
    if subprocess.run([*patch[:2], "--check", *patch[2:]], cwd=root, capture_output=True, check=False).returncode == 0:
        subprocess.run(patch, cwd=root, check=True, capture_output=True)
    return root / "scripts" / "hooks" / "pre-push"


@pytest.fixture
def checkout(tmp_path):
    where = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "0.9dev", str(where)], check=True)
    commit(where, "README.md")
    return where


def push(hook: Path, where: Path, ref: str, sha: str, base: str) -> subprocess.CompletedProcess:
    installed = Path(git(where, "rev-parse", "--path-format=absolute", "--git-common-dir")) / "hooks" / "pre-push"
    installed.write_text("#!/bin/sh\nexec sh " + shlex.quote(hook.as_posix()) + ' "$@"\n', encoding="utf-8")
    installed.chmod(installed.stat().st_mode | 0o111)
    source = installed.parent.parent / "push-input"
    source.write_text(f"refs/heads/{ref} {sha} refs/heads/{ref} {base}\n", encoding="utf-8")
    return subprocess.run(["git", "hook", "run", "--to-stdin=" + str(source), "pre-push", "--",
                           "origin", "https://example.invalid/x.git"], cwd=where, text=True, capture_output=True,
                          env={**os.environ, "PYTHON": Path(sys.executable).as_posix(), "CLAUDECODE": "1"})


def test_an_existing_promote_branch_moves_forward(hooks, checkout):
    first = git(checkout, "rev-parse", "HEAD")
    second = commit(checkout, "later.txt")
    done = push(hooks, checkout, "promote/2026-10-09", second, first)
    assert done.returncode == 0, done.stderr


def test_a_rewrite_of_a_promote_branch_is_refused(hooks, checkout):
    first = git(checkout, "rev-parse", "HEAD")
    git(checkout, "checkout", "-q", "-b", "other")
    other = commit(checkout, "diverged.txt")
    git(checkout, "checkout", "-q", "0.9dev")
    mine = commit(checkout, "mine.txt")
    assert first != other != mine
    assert push(hooks, checkout, "promote/2026-10-09", mine, other).returncode != 0


def test_a_promote_branch_whose_remote_commit_is_unknown_here_is_refused(hooks, checkout):
    assert push(hooks, checkout, "promote/x", git(checkout, "rev-parse", "HEAD"), "1" * 40).returncode != 0


def test_a_new_promote_branch_still_goes_through(hooks, checkout):
    assert push(hooks, checkout, "promote/new", git(checkout, "rev-parse", "HEAD"), ZERO).returncode == 0


def test_main_and_a_work_branch_stay_refused(hooks, checkout):
    first = git(checkout, "rev-parse", "HEAD")
    second = commit(checkout, "later.txt")
    assert push(hooks, checkout, "main", second, first).returncode != 0
    assert push(hooks, checkout, "split-something", second, first).returncode != 0
    assert push(hooks, checkout, "work/promote/x", second, first).returncode != 0
