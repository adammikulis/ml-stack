"""The worktree registry's git call, attacked: a root or an argument that is text, never a command."""

from __future__ import annotations

import pytest

from ml_stack import trees


@pytest.mark.parametrize("root", ["$(touch pwned)", "x; touch pwned", "`touch pwned`", "a\nb", "--upload-pack=touch pwned"])
def test_a_hostile_root_is_only_a_path_and_runs_nothing(tmp_path, monkeypatch, root):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError):
        trees.git(root, "rev-parse", "--git-dir")
    assert not (tmp_path / "pwned").exists()


def test_a_stalled_git_is_an_error_not_a_hang(tmp_path, monkeypatch):
    import subprocess

    def stall(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(trees.subprocess, "run", stall)
    with pytest.raises(RuntimeError, match="did not answer"):
        trees.git(tmp_path, "status")
