"""A budget is a ratchet: over it fails, and under it fails until --update locks the gain in."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import gates  # noqa: E402

BUDGETS = REPO / "budgets.json"
FOUND = gates.run(REPO)
ALLOWED: dict[str, int] = json.loads(BUDGETS.read_text(encoding="utf-8"))


def _sites(name: str) -> str:
    return "\n".join(f"    {f.path}:{f.line}  {f.detail}" for f in FOUND[name][:10])


def _runnable(checker) -> None:
    reason = gates.skipped(checker)
    if reason:
        pytest.skip(f"{checker.NAME}: {reason}")


@pytest.mark.parametrize("checker", gates.checkers(), ids=lambda c: c.NAME)
def test_metric_is_within_its_budget(checker) -> None:
    _runnable(checker)
    name = checker.NAME
    budget = ALLOWED[name]
    actual = len(FOUND[name])
    owner = checker.OWNER or "no owner yet"
    assert actual <= budget, (
        f"{name}: {actual} sites, {budget} allowed. {checker.describe()}\n"
        f"owned by {owner}\n{_sites(name)}\n"
        f"    scripts/budgets --show {name}"
    )


@pytest.mark.parametrize("checker", gates.checkers(), ids=lambda c: c.NAME)
def test_budget_matches_the_tree(checker) -> None:
    _runnable(checker)
    name = checker.NAME
    budget = ALLOWED[name]
    actual = len(FOUND[name])
    assert actual >= budget, (
        f"{name}: {actual} sites but {budget} allowed -- the budget is stale. "
        f"Run scripts/budgets --update so the number can only fall from here."
    )


def test_every_checker_has_a_budget() -> None:
    names = {c.NAME for c in gates.checkers()}
    missing = sorted(names - set(ALLOWED))
    assert not missing, (
        f"no budget for {', '.join(missing)} -- run scripts/budgets --update"
    )


def test_no_budget_without_a_checker() -> None:
    names = {c.NAME for c in gates.checkers()}
    extra = sorted(set(ALLOWED) - names)
    assert not extra, (
        f"budgets.json names {', '.join(extra)} with no checker -- run scripts/budgets --update"
    )


def test_a_checker_that_cannot_run_here_names_a_metric_with_a_budget() -> None:
    for name, reason in gates.unrunnable().items():
        assert name in ALLOWED, f"{name} has no budget"
        assert reason, f"{name} skipped with no reason"


def test_the_hook_counts_what_a_commit_deletes(tmp_path):
    """A commit that moves code out of one file and into another is not read as addition."""
    import subprocess

    def run(*a):
        return subprocess.run(["git", *a], cwd=tmp_path, capture_output=True,
                              text=True, check=True)

    hook = REPO / "scripts" / "hooks" / "budgets"
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    run("config", "user.email", "nobody@example.invalid")
    run("config", "user.name", "A Tester")
    src = tmp_path / "src" / "ml_stack"
    src.mkdir(parents=True)
    (src / "leaving.py").write_text('def a():\n    print("one")\n    print("two")\n')
    (src / "staying.py").write_text("def b():\n    return 1\n")
    (tmp_path / "budgets.json").write_text('{"print-calls": 2}\n')
    run("add", "-A")
    run("commit", "-qm", "before")

    run("rm", "-q", "src/ml_stack/leaving.py")
    (src / "staying.py").write_text('def b():\n    print("moved")\n    return 1\n')
    run("add", "src/ml_stack/staying.py")

    done = subprocess.run([sys.executable, str(hook)], cwd=tmp_path,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
