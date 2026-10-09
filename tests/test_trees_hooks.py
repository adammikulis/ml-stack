"""The hooks' side of the worktree registry: a started tree is claimed, a stopped tree's debt is posted at once."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
from test_trees import commit, git, main, tree  # noqa: F401  (the repository fixture and helpers)

from ml_stack.workspace import trees

REPO = Path(__file__).resolve().parent.parent


def as_agent(name: str) -> dict:
    """The environment a hook runs a workspace command in."""
    return {**os.environ, "ML_STACK_WORKSPACE_AGENT": name}


@pytest.fixture
def board(tmp_path, monkeypatch):
    """A stand-in `ml-stack-workspace` on PATH that records every call, and the hook module."""
    calls = tmp_path / "calls.txt"
    stub = tmp_path / "bin" / "ml-stack-workspace"
    stub.parent.mkdir()
    stub.write_text(f'#!/bin/sh\necho "$@" >> {calls}\n')
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "hooks"))
    import tree_watch
    return tree_watch, calls


def test_the_hooks_register_a_started_tree_and_post_a_stopped_trees_debt(main, board):  # noqa: F811
    tree_watch, calls = board
    trees.set_lead(main, "lead")
    path = main.parent / "iso"
    git(main, "worktree", "add", "-q", "-b", "iso", str(path))
    tree_watch.started("SubagentStart", str(path), "worker-z", "a test")
    assert trees.rows(main, time.time())[0]["owner"] == "worker-z"
    commit(path, "f1")
    tree_watch.stopped("SubagentStop", str(path), as_agent("worker-z"))
    said = calls.read_text()
    assert "announce milestone ORPHAN iso" in said and "dm lead ORPHAN iso" in said
    assert trees.orphans(trees.rows(main, time.time()))[0]["owner"] == "worker-z"


def test_the_hooks_say_nothing_for_a_stopped_tree_with_no_debt(main, board):  # noqa: F811
    tree_watch, calls = board
    path = tree(main, "clean", owner="worker-y", now=time.time())
    tree_watch.stopped("SubagentStop", str(path), as_agent("worker-y"))
    assert not calls.exists()
    assert tree_watch.check("SessionStart", str(main), as_agent("lead"), lead="lead") == ""


def test_the_post_commit_check_only_looks_at_the_tree_committed_in(main, board):  # noqa: F811
    tree_watch, calls = board
    trees.set_lead(main, "lead")
    path = tree(main, "big", owner="worker-x", now=time.time())
    other = tree(main, "other", owner="worker-w", now=time.time())
    for i in range(10):
        commit(path, f"f{i}")
        commit(other, f"g{i}")
    tree_watch.check("post-commit", str(path), as_agent("worker-x"), here=True)
    said = calls.read_text()
    assert "dm worker-x" in said and "big" in said and "other" not in said
    assert sys.modules["tree_watch"] is tree_watch
