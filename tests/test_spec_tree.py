"""A draft tree's shape, the budget that trims one, and which path an acceptance rule keeps."""

from __future__ import annotations

import numpy as np
import pytest

from ml_stack.spec.accept import Rule, walk
from ml_stack.spec.tree import Candidates, Tree, budgeted, chain


def test_a_node_knows_its_depth_ancestors_and_children() -> None:
    tree = Tree([10, 11, 12, 13, 14], [-1, 0, 0, 1, 3])
    assert tree.depth == [0, 1, 1, 2, 3]
    assert tree.ancestors[4] == [4, 3, 1, 0]
    assert tree.ancestors[2] == [2, 0, -1, -1]
    assert tree.children[0] == [1, 2]
    assert tree.path_to(4) == [0, 1, 3, 4]


def test_a_node_before_its_parent_is_refused() -> None:
    with pytest.raises(ValueError):
        Tree([1, 2, 3], [-1, 2, 0])


def test_the_budget_stops_where_another_node_costs_more_than_it_is_worth() -> None:
    found = Candidates([0, 1, 2, 3, 4], [-1, 0, 1, 2, 0], [1.0, 0.9, 0.8, 0.1, 0.05])
    flat = budgeted(found, {1: 1.0, 32: 1.0}, draft_cost=0.0, max_nodes=32)
    assert flat.n == 5
    steep = budgeted(found, {1: 1.0, 2: 1.5, 3: 2.0, 4: 4.0, 5: 8.0}, draft_cost=0.0,
                     max_nodes=32)
    assert steep.tokens == [0, 1, 2]


def test_the_budget_never_keeps_a_node_without_its_parent() -> None:
    found = Candidates([0, 1, 2], [-1, 0, 1], [1.0, 0.1, 0.9])
    tree = budgeted(found, {1: 1.0, 2: 1.0, 3: 100.0}, draft_cost=0.0, max_nodes=2)
    assert tree.tokens == [0]


def test_lossless_follows_the_draw_and_the_bonus_is_the_draw_that_missed() -> None:
    tree = Tree([7, 8, 9, 5, 6], [-1, 0, 0, 1, 1])
    kept = walk(tree, [8, 6, 1, 1, 2], None, None, Rule())
    assert kept.path == [0, 1, 4]
    assert kept.bonus == 2
    assert kept.forced == 0


def test_ratio_takes_a_missed_child_only_when_little_mass_moves_per_token_gained() -> None:
    tree = chain([7, 8, 9, 10])
    ids = np.array([[3, 8], [9, 1], [10, 1], [4, 1]])
    drawn = [3, 9, 10, 4]
    lots = np.array([[0.2, 0.8], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    kept = walk(tree, drawn, ids, lots, Rule.parse("ratio:theta=0.3"))
    assert kept.path == [0, 1, 2, 3]
    assert kept.forced == 1
    little = np.array([[0.95, 0.05], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert walk(tree, drawn, ids, little, Rule.parse("ratio:theta=0.3")).path == [0]
    assert walk(tree, drawn, ids, lots, Rule.parse("lossless")).path == [0]


def test_a_rule_names_only_settings_it_has() -> None:
    assert Rule.parse("ratio:theta=0.2").theta == 0.2
    with pytest.raises(ValueError):
        Rule.parse("ratio:eps=1")
    with pytest.raises(ValueError):
        Rule.parse("cov")
