"""The git hook that lets an agent push the development branch and nothing else."""

import os
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / "scripts" / "hooks" / "pre-push"
ZERO = "0" * 40


def git(where: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(where), *args], check=True, text=True,
                          capture_output=True, env=env).stdout.strip()


def commit(where: Path, name: str, text: str) -> str:
    (where / name).parent.mkdir(parents=True, exist_ok=True)
    (where / name).write_text(text)
    git(where, "add", name)
    git(where, "commit", "-q", "-m", name)
    return git(where, "rev-parse", "HEAD")


@pytest.fixture
def checkout(tmp_path):
    """A repository whose primary checkout is on the development branch `0.9dev`."""
    where = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "0.9dev", str(where)], check=True)
    commit(where, "README.md", "x\n")
    return where


def push(where: Path, *refs: str, sha: str = "", base: str = ZERO,
         **env: str) -> subprocess.CompletedProcess:
    sha = sha or git(where, "rev-parse", "HEAD")
    lines = "".join(f"refs/heads/{r} {sha} refs/heads/{r} {base}\n" for r in refs)
    return subprocess.run(
        [str(HOOK), "origin", "https://example.invalid/x.git"], cwd=where,
        text=True, capture_output=True, input=lines, env={**os.environ, **env})


def test_an_agent_pushes_the_development_branch(checkout):
    done = push(checkout, "0.9dev", CLAUDECODE="1")
    assert done.returncode == 0, done.stderr


def test_an_agents_push_of_main_is_refused(checkout):
    done = push(checkout, "main", CLAUDECODE="1")
    assert done.returncode != 0
    assert "refused: refs/heads/main" in done.stderr
    assert "git push origin 0.9dev" in done.stderr


def test_main_rides_along_with_the_development_branch_and_is_still_refused(checkout):
    done = push(checkout, "0.9dev", "main", CLAUDECODE="1")
    assert done.returncode != 0
    assert "refs/heads/main" in done.stderr and "refs/heads/0.9dev" not in done.stderr


def test_an_agents_push_of_a_work_branch_is_refused(checkout):
    assert push(checkout, "split-something", CLAUDECODE="1").returncode != 0


def test_an_agent_cannot_delete_the_development_branch(checkout):
    assert push(checkout, "0.9dev", sha=ZERO, CLAUDECODE="1").returncode != 0


def test_main_is_refused_even_when_the_primary_checkout_is_on_it(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    commit(tmp_path, "README.md", "x\n")
    assert push(tmp_path, "main", CLAUDECODE="1").returncode != 0


def test_a_person_is_not_stopped(checkout):
    """No terminal sets CLAUDECODE, and neither does a GUI client, so the owner's own
    push meets nothing -- the one push this hook must never be in the way of."""
    done = push(checkout, "main", "split-something", CLAUDECODE="")
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""


def test_the_installer_wires_it_up():
    installer = HOOK.parent.parent / "install-hooks.sh"
    assert "pre-push pre-push" in installer.read_text()


def test_an_agents_push_of_python_that_does_not_parse_is_refused(checkout):
    base = git(checkout, "rev-parse", "HEAD")
    commit(checkout, "pkg/broken.py", "def f(:\n")
    done = push(checkout, "0.9dev", base=base, CLAUDECODE="1")
    assert done.returncode != 0
    assert "pkg/broken.py" in done.stderr


def test_an_agents_push_of_a_data_file_is_refused(checkout):
    base = git(checkout, "rev-parse", "HEAD")
    commit(checkout, "store.lbug", "x")
    done = push(checkout, "0.9dev", base=base, CLAUDECODE="1")
    assert done.returncode != 0
    assert "store.lbug" in done.stderr


def test_a_merged_worktree_blocks_the_development_branch_until_removed_or_locked(checkout):
    tree = checkout.parent / "done"
    git(checkout, "worktree", "add", "-q", "-b", "done-work", str(tree))
    commit(tree, "work.py", "z = 3\n")
    git(checkout, "merge", "-q", "--ff-only", "done-work")
    done = push(checkout, "0.9dev", CLAUDECODE="1")
    assert done.returncode != 0
    assert f"git worktree remove {tree}" in done.stderr
    git(checkout, "worktree", "lock", str(tree))
    assert push(checkout, "0.9dev", CLAUDECODE="1").returncode == 0


def test_a_worktree_with_its_own_commits_does_not_block(checkout):
    tree = checkout.parent / "live"
    git(checkout, "worktree", "add", "-q", "-b", "live-work", str(tree))
    commit(tree, "new.py", "y = 2\n")
    assert push(checkout, "0.9dev", CLAUDECODE="1").returncode == 0


def test_a_fresh_worktree_with_no_commits_yet_does_not_block(checkout):
    git(checkout, "worktree", "add", "-q", "-b", "just-started", str(checkout.parent / "fresh"))
    assert push(checkout, "0.9dev", CLAUDECODE="1").returncode == 0
