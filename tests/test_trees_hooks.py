"""The hooks' side of the worktree registry: a started tree is claimed, a stopped tree's debt is posted at once."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
from test_trees import commit, git, make_repo, tree

from poolhouse import trees

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def main(tmp_path):
    return make_repo(tmp_path)


def as_agent(name: str) -> dict:
    """The environment a hook runs a workspace command in."""
    return {**os.environ, "POOLHOUSE_WORKSPACE_AGENT": name}


@pytest.fixture
def board(tmp_path, monkeypatch):
    """A stand-in `poolhouse-workspace` on PATH that records every call, and the hook module."""
    calls = tmp_path / "calls.txt"
    stub = tmp_path / "bin" / "poolhouse-workspace"
    stub.parent.mkdir()
    stub.write_text(f'#!/bin/sh\necho "$@" >> {calls}\n')
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "hooks"))
    import tree_watch
    monkeypatch.setattr(tree_watch, "DETACH", False)
    return tree_watch, calls


def test_the_hooks_register_a_started_tree_and_post_a_stopped_trees_debt(main, board):
    tree_watch, calls = board
    trees.set_lead(main, "lead")
    path = main.parent / "iso"
    git(main, "worktree", "add", "-q", "-b", "iso", str(path))
    tree_watch.started("SubagentStart", str(path), "worker-z", "a test")
    assert trees.rows(main, time.time())[0]["owner"] == "worker-z"
    commit(path, "f1")
    tree_watch.stopped("SubagentStop", str(path), as_agent("worker-z"))
    said = calls.read_text()
    assert "announce milestone tree iso" in said and "dm lead tree iso" in said and "waiting to land" in said
    assert "ORPHAN" not in said
    assert trees.waiting(trees.rows(main, time.time()))[0]["owner"] == "worker-z"


def test_the_hooks_say_nothing_for_a_stopped_tree_with_no_debt(main, board):
    tree_watch, calls = board
    path = tree(main, "clean", owner="worker-y", now=time.time())
    tree_watch.stopped("SubagentStop", str(path), as_agent("worker-y"))
    assert not calls.exists()
    assert tree_watch.check("SessionStart", str(main), as_agent("lead"), lead="lead") == ""


def test_the_post_commit_check_only_looks_at_the_tree_committed_in(main, board):
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


def test_delivery_to_a_slow_board_does_not_hold_up_the_hook(main, board, monkeypatch):
    tree_watch, calls = board
    monkeypatch.setattr(tree_watch, "DETACH", True)
    stub = calls.parent / "bin" / "poolhouse-workspace"
    stub.write_text(f'#!/bin/sh\nsleep 2\necho "$@" >> {calls}\n')
    trees.set_lead(main, "lead")
    path = tree(main, "big", owner="worker-x", now=time.time())
    for i in range(10):
        commit(path, f"f{i}")
    started = time.monotonic()
    tree_watch.check("post-commit", str(path), as_agent("worker-x"), here=True)
    assert time.monotonic() - started < 1.5
    for _ in range(100):
        if calls.exists() and "dm worker-x" in calls.read_text():
            break
        time.sleep(0.2)
    assert "dm worker-x" in calls.read_text()
