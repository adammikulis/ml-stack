"""Test results as evidence: `scripts/test` records each pytest run against the hash of the tree
it ran on, so a green gate on a tree is something the tool wrote. Real git, real pytest."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from ml_stack.activity import gate
from tests.activity_support import entries, person, ring

__all__ = ["person", "ring"]
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "test"


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                    *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    (root / "test_a.py").write_text("def test_one():\n    assert True\n\ndef test_two():\n    assert True\n")
    return root


def script():
    loader = importlib.machinery.SourceFileLoader("scripts_test_runner", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def pytest_in(_root: Path) -> list[str]:
    return [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_a.py"]


def test_the_tree_hash_follows_the_content_and_ignores_the_real_index(repo):
    first = gate.tree_hash(repo)
    assert len(first) == 40 and gate.tree_hash(repo) == first
    (repo / "test_a.py").write_text("def test_one():\n    assert False\n")
    assert gate.tree_hash(repo) != first
    assert subprocess.run(["git", "-C", str(repo), "status", "--short"], capture_output=True,
                          text=True).stdout.startswith("??")


def test_the_tree_hash_is_empty_outside_a_checkout(tmp_path):
    assert gate.tree_hash(tmp_path / "nowhere") == ""


def test_a_green_run_is_recorded_against_its_tree_and_found_as_evidence(person, repo):
    runner = script()
    runner.ROOT = repo
    assert runner.run(pytest_in(repo), tier="full") == 0
    [e] = [x for x in entries() if x.kind == "test.result"]
    tree = gate.tree_hash(repo)
    assert (e.outcome, e.refs["tree"], e.refs["tier"], e.subject) == ("pass", tree, "full", f"tree:{tree[:12]}")
    assert (e.meta["passed"], e.meta["failed"], e.meta["exit"]) == (2, 0, 0) and e.meta["seconds"] >= 0
    assert gate.evidence(tree, "full").id == e.id
    assert gate.evidence(tree, "slow") is None


def test_a_red_run_is_not_evidence_and_a_changed_tree_has_none(person, repo):
    runner = script()
    runner.ROOT = repo
    (repo / "test_a.py").write_text("def test_one():\n    assert False\n")
    assert runner.run(pytest_in(repo), tier="full") == 1
    [e] = [x for x in entries() if x.kind == "test.result"]
    assert (e.outcome, e.meta["failed"], e.meta["passed"]) == ("fail", 1, 0)
    assert gate.evidence(e.refs["tree"]) is None
    (repo / "test_a.py").write_text("def test_one():\n    assert True\n")
    assert gate.evidence(gate.tree_hash(repo)) is None


def test_the_same_command_on_the_same_tree_has_the_same_command_hash(person, repo):
    runner = script()
    runner.ROOT = repo
    runner.run(pytest_in(repo), tier="full")
    runner.run(pytest_in(repo), tier="full")
    one, two = [x for x in entries() if x.kind == "test.result"]
    assert one.refs["command"] == two.refs["command"] and one.refs["tree"] == two.refs["tree"]
