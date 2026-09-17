"""Drafters: what proposes a tree of tokens for the target to verify in one pass.

Every drafter answers the same five calls. ``prefill`` reads the prompt, whose last token is
the first root, with the target's hidden states for the tokens before it; ``draft`` proposes
a tree under a root; ``accept`` reads the path the target kept; ``state`` and ``restore``
snapshot the drafter's context for a session that reuses a prefix. ``taps`` names the target
layers whose outputs the drafter reads, empty for the last layer's pre-norm hidden state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import mlx.core as mx

from ml_stack.spec.tree import Tree

__all__ = ["Budget", "Drafter"]


@dataclass(frozen=True)
class Budget:
    """How big a tree may be: the node cap and the measured cost of verifying n nodes."""

    max_nodes: int = 32
    cost: Mapping[int, float] = field(default_factory=lambda: {1: 1.0})


class Drafter(Protocol):
    taps: tuple[int, ...]

    def prefill(self, tokens: Sequence[int], hidden: mx.array) -> None: ...

    def draft(self, root: int, hidden: mx.array) -> Tree: ...

    def accept(self, tokens: Sequence[int], hidden: mx.array) -> None: ...

    def state(self) -> object: ...

    def restore(self, state: object) -> None: ...
