"""The git hook that lets an agent push the development branch and nothing else."""

import os
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / "scripts" / "hooks" / "pre-push"
ZERO = "0" * 40


@pytest.fixture
def checkout(tmp_path):
    """A repository whose primary checkout is on the development branch `0.9dev`."""
    subprocess.run(["git", "init", "-q", "-b", "0.9dev", str(tmp_path)], check=True)
    return tmp_path


def push(where: Path, *refs: str, sha: str = "deadbee",
         **env: str) -> subprocess.CompletedProcess:
    lines = "".join(f"refs/heads/{r} {sha} refs/heads/{r} f00ba12\n" for r in refs)
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
