"""Judging whether a destination sits inside a repository, so nothing real is committed."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ml_stack.paths import inside_a_repo, repo_root


def repo(where: Path) -> Path:
    where.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=where, check=True, capture_output=True)
    return where


def test_a_path_inside_a_git_worktree_is_inside_a_repo(tmp_path):
    root = repo(tmp_path / "project")
    (root / "out").mkdir()
    assert inside_a_repo(root)
    assert inside_a_repo(root / "out")
    assert inside_a_repo(root / "out" / "runs.jsonl"), "a file not written yet, by its parent"


def test_a_path_outside_any_repo_is_not(tmp_path):
    repo(tmp_path / "project")
    (tmp_path / "backups").mkdir()
    assert not inside_a_repo(tmp_path / "backups")
    assert not inside_a_repo(tmp_path / "backups" / "later" / "export.json"), \
        "nothing exists below backups/, so backups/ is what is judged"


def test_a_dot_git_file_counts_as_much_as_a_directory(tmp_path):
    """A linked worktree or a submodule keeps `.git` as a file pointing at the real one."""
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / ".git").write_text("gitdir: /somewhere/else/.git/worktrees/linked\n")
    assert inside_a_repo(linked / "deep" / "file.md")


def test_a_symlink_into_a_repo_is_inside_it(tmp_path):
    root = repo(tmp_path / "project")
    (root / "exports").mkdir()
    link = tmp_path / "shortcut"
    link.symlink_to(root / "exports")
    assert inside_a_repo(link / "runs.jsonl")


def test_a_tilde_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    repo(tmp_path / "home-repo")
    assert inside_a_repo("~/home-repo/anything")
    assert not inside_a_repo("~/elsewhere")


def test_the_root_is_named_so_a_refusal_can_say_which_repository(tmp_path):
    root = repo(tmp_path / "project")
    (root / "docs").mkdir()
    assert repo_root(root / "docs" / "runs.json") == root.resolve()
    assert repo_root(tmp_path / "elsewhere" / "runs.json") is None
