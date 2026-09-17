"""Prompt lookup: continuations already seen after the same few tokens, as a small tree."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from ml_stack.spec.drafters import Budget
from ml_stack.spec.tree import Tree

__all__ = ["NgramDrafter"]


class NgramDrafter:
    """Up to ``branch`` continuations per level, ``depth`` levels, matched on ``n`` tokens."""

    taps: tuple[int, ...] = ()

    def __init__(self, budget: Budget, n: int = 3, depth: int = 8, branch: int = 2) -> None:
        self.budget, self.n, self.depth, self.branch = budget, n, depth, branch
        self.history: list[int] = []

    def prefill(self, tokens: Sequence[int], hidden: mx.array) -> None:
        self.history = list(tokens[:-1])

    def accept(self, tokens: Sequence[int], hidden: mx.array) -> None:
        self.history.extend(tokens)

    def state(self) -> object:
        return len(self.history)

    def restore(self, state: object) -> None:
        del self.history[int(state):]  # type: ignore[call-overload]

    def _after(self, context: tuple[int, ...]) -> list[int]:
        found: list[int] = []
        width = len(context)
        seen = self.history
        for i in range(len(seen) - width - 1, -1, -1):
            if tuple(seen[i:i + width]) == context and seen[i + width] not in found:
                found.append(seen[i + width])
                if len(found) >= self.branch:
                    break
        return found

    def draft(self, root: int, hidden: mx.array) -> Tree:
        tokens, parent = [root], [-1]
        start = (*self.history[-(self.n - 1):], root) if self.n > 1 else (root,)
        frontier = [(0, tuple(start))]
        for _ in range(self.depth):
            grown = []
            for node, context in frontier:
                for token in self._after(context):
                    if len(tokens) >= self.budget.max_nodes:
                        break
                    tokens.append(token)
                    parent.append(node)
                    grown.append((len(tokens) - 1, (*context, token)[-self.n:]))
            frontier = grown
            if not frontier:
                break
        return Tree(tokens, parent)
