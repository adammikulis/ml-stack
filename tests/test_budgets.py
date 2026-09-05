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


@pytest.mark.parametrize("checker", gates.checkers(), ids=lambda c: c.NAME)
def test_metric_is_within_its_budget(checker) -> None:
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
