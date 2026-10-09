"""A copied checkout is a working one: its linked worktree points at the copy and the template stays clean."""

from __future__ import annotations

from git_template import copy_checkout

from poolhouse.net import git


def _build(root):
    primary = root / "main"
    primary.mkdir()
    git.run(["init", "-q", str(primary)])
    (primary / "a.txt").write_text("one\n")
    git.run(["add", "a.txt"], cwd=primary)
    git.run(["-c", "user.name=T", "-c", "user.email=t@example.invalid", "commit", "-qm", "base"], cwd=primary)
    git.run(["worktree", "add", "-q", "-b", "side", str(root / "linked")], cwd=primary)


def test_a_copied_checkout_has_its_linked_worktree_pointing_at_the_copy(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    copy_checkout("pair", _build, first, "main", "linked")
    copy_checkout("pair", _build, second, "main", "linked")
    listed = git.run(["worktree", "list", "--porcelain"], cwd=second / "main").stdout
    assert str(second / "linked") in listed and str(first / "linked") not in listed
    (second / "linked" / "b.txt").write_text("two\n")
    git.run(["add", "b.txt"], cwd=second / "linked")
    git.run(["-c", "user.name=T", "-c", "user.email=t@example.invalid", "commit", "-qm", "more"], cwd=second / "linked")
    other = git.run(["log", "--oneline", "side"], cwd=first / "main").stdout
    assert "more" not in other
