"""Portable hook installation and Git invocation."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_installed_hook_runs_in_git_and_preserves_foreign_hook(tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    shipped = repo / "scripts" / "hooks"
    shipped.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/hooks/pre-push", shipped / "pre-push")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "CLAUDECODE": ""}
    subprocess.run(["git", "add", "."], cwd=repo, env=env, check=True)
    subprocess.run(["git", "commit", "-qm", "chore: hooks"], cwd=repo, env=env, check=True)
    foreign = repo / ".git/hooks/commit-msg"
    foreign.write_text("#!/bin/sh\nexit 0\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/install-hooks.py")], cwd=repo, check=True)
    assert foreign.read_text() == "#!/bin/sh\nexit 0\n"
    command = ["git", "hook", "run", "pre-push", "--", "origin", "unused"]
    clean = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True)
    assert clean.returncode == 0, clean.stderr
    (repo / "pending").write_text("pending")
    dirty = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True)
    assert dirty.returncode != 0
    assert "uncommitted changes" in dirty.stderr
