"""A draft tree: tokens hanging off the last committed token, and the budget that trims one."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["Candidates", "Tree", "budgeted", "chain"]


@dataclass
class Tree:
    """Nodes in topological order; node 0 is the root, the token not yet in the cache."""

    tokens: list[int]
    parent: list[int]
    values: list[float] = field(default_factory=list)
    depth: list[int] = field(init=False)
    ancestors: list[list[int]] = field(init=False)
    children: list[list[int]] = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.tokens)
        if len(self.parent) != n or not n or self.parent[0] != -1:
            raise ValueError("a tree needs one parent per token and a root whose parent is -1")
        self.depth = [0] * n
        self.children = [[] for _ in range(n)]
        for i in range(1, n):
            if not 0 <= self.parent[i] < i:
                raise ValueError(f"node {i} comes before its parent {self.parent[i]}")
            self.depth[i] = self.depth[self.parent[i]] + 1
            self.children[self.parent[i]].append(i)
        reach = max(self.depth) + 1
        self.ancestors = []
        for i in range(n):
            row, j = [], i
            for _ in range(reach):
                row.append(j)
                j = self.parent[j] if j >= 0 else -1
            self.ancestors.append(row)

    @property
    def n(self) -> int:
        return len(self.tokens)

    @property
    def max_depth(self) -> int:
        return max(self.depth)

    def path_to(self, node: int) -> list[int]:
        """The node indices from the root down to ``node``."""
        return [j for j in reversed(self.ancestors[node]) if j >= 0]


@dataclass
class Candidates:
    """Every node a drafter proposed, before the budget picks which ones are verified."""

    tokens: list[int]
    parent: list[int]
    values: list[float]


def _cost_at(cost: Mapping[int, float], n: int) -> float:
    sizes = sorted(cost)
    for size in sizes:
        if size >= n:
            return cost[size]
    return cost[sizes[-1]] * n / sizes[-1]


def budgeted(found: Candidates, cost: Mapping[int, float], *, draft_cost: float,
             max_nodes: int) -> Tree:
    """The value-sorted prefix of ``found`` that maximises expected tokens per unit of cost.

    ``cost[n]`` is verifying n nodes relative to one; a node is kept only with its parent.
    """
    order = sorted(range(1, len(found.tokens)), key=lambda i: (-found.values[i], i))
    best_n, best, total = 1, 1.0 / (_cost_at(cost, 1) + draft_cost), 0.0
    for j, i in enumerate(order[:max_nodes - 1]):
        total += found.values[i]
        score = (1.0 + total) / (_cost_at(cost, j + 2) + draft_cost)
        if score > best:
            best_n, best = j + 2, score
    chosen = set(order[:best_n - 1])
    changed = True
    while changed:
        changed = False
        for i in sorted(chosen):
            if found.parent[i] > 0 and found.parent[i] not in chosen:
                chosen.discard(i)
                changed = True
    remap, tokens, parent, values = {0: 0}, [found.tokens[0]], [-1], [1.0]
    for i in sorted(chosen):
        remap[i] = len(tokens)
        tokens.append(found.tokens[i])
        parent.append(remap[found.parent[i]])
        values.append(found.values[i])
    return Tree(tokens, parent, values)


def chain(tokens: Sequence[int]) -> Tree:
    """A tree with no branches: each token the child of the one before."""
    return Tree(list(tokens), [-1, *range(len(tokens) - 1)])
