"""One forward pass over a whole draft tree, and the commit that keeps the accepted path."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from ml_stack.spec import attention, deltanet
from ml_stack.spec.layout import Layout
from ml_stack.spec.tree import Tree

__all__ = ["TreePass", "TreeVerifier"]

#: dispatch the graph every this many layers so the GPU starts while the rest is built
ASYNC_EVERY = 8


@dataclass
class TreePass:
    """What every layer of one tree pass shares, and what its layers leave to commit.

    ``attention`` collects the attention caches the pass appended to, compacted together on
    commit; each of ``commits`` is called with the accepted path's node indices.
    """

    tree: Tree
    offset: int
    ancestors: mx.array
    positions: mx.array
    mask: mx.array
    attention: list[Any] = field(default_factory=list)
    commits: list[Callable[[mx.array], None]] = field(default_factory=list)
    taps: dict[tuple[int, int], mx.array] = field(default_factory=dict)

    @classmethod
    def over(cls, tree: Tree, offset: int) -> TreePass:
        """The pass for ``tree`` hanging off ``offset`` cached tokens."""
        return cls(tree, offset, mx.array(tree.ancestors, mx.int32),
                   mx.array([offset + d for d in tree.depth], mx.int32),
                   attention.tree_mask(tree.ancestors, offset))

    def conv_taps(self, width: int, dilation: int = 1) -> mx.array:
        """Rows into ``[prefix rows | tree rows]`` for a causal convolution along each path."""
        key = (width, dilation)
        if key not in self.taps:
            self.taps[key] = deltanet.conv_taps(self.tree.depth, self.tree.ancestors, width,
                                                dilation)
        return self.taps[key]


class TreeVerifier:
    """Next-token logits for every node of a tree hanging off ``cache``, then a commit.

    ``taps`` are layer indices whose outputs, as the layout taps them, are concatenated into
    ``fused`` on every pass, for a drafter that reads the target's residual stream.
    """

    def __init__(self, layout: Layout, cache: list[Any], taps: Sequence[int] = ()) -> None:
        self.layout = layout
        self.cache = cache
        self.taps = tuple(taps)
        self.fused: mx.array | None = None
        self._pass: TreePass | None = None

    def prefill(self, tokens: Sequence[int], chunk: int = 2048) -> mx.array:
        """Run ``tokens`` into the cache; returns their residual states [T, D]."""
        hidden, fused = [], []
        for start in range(0, len(tokens), chunk):
            ids = mx.array(list(tokens[start:start + chunk]), mx.uint32)[None]
            out, tapped = self.layout.prefill(ids, self.cache, self.taps)
            mx.eval(out, *tapped)
            hidden.append(out[0])
            if tapped:
                fused.append(mx.concatenate([t[0] for t in tapped], axis=-1))
        self.fused = mx.concatenate(fused, axis=0) if fused else None
        return mx.concatenate(hidden, axis=0)

    @property
    def offset(self) -> int:
        """Tokens already in the caches."""
        return self.layout.offset(self.cache)

    def forward(self, tree: Tree) -> tuple[mx.array, mx.array]:
        """``(logits [N, V], residual [N, D])`` for every node of ``tree``."""
        layout = self.layout
        walk = TreePass.over(tree, self.offset)
        self._pass = walk
        hidden = layout.enter(mx.array(tree.tokens, mx.uint32))
        tapped = []
        wanted = set(self.taps)
        for index in range(layout.layer_count):
            hidden = layout.block(index, hidden, self.cache[index], walk)
            if index in wanted:
                tapped.append(layout.tap(index, hidden))
            if index % ASYNC_EVERY == ASYNC_EVERY - 1:
                mx.async_eval(hidden)
        self.fused = mx.concatenate(tapped, axis=-1) if tapped else None
        return layout.logits(hidden), hidden

    def commit(self, path: Sequence[int]) -> None:
        """Keep the root-to-node ``path`` (node indices, starting at 0) in every cache."""
        walk = self._pass
        if walk is None:
            raise RuntimeError("commit follows a forward pass")
        rows = mx.array(list(path), mx.int32)
        attention.compact(walk.attention, walk.offset, rows)
        for keep in walk.commits:
            keep(rows)
        self._pass = None
