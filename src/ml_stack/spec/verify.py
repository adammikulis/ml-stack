"""One forward pass over a whole draft tree, and the commit that keeps the accepted path."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx

from ml_stack.spec import attention, deltanet
from ml_stack.spec.layout import Layout
from ml_stack.spec.tree import Tree

__all__ = ["TreeVerifier"]

#: dispatch the graph every this many layers so the GPU starts while the rest is built
ASYNC_EVERY = 8


class TreeVerifier:
    """Next-token logits for every node of a tree hanging off ``cache``, then a commit.

    ``taps`` are layer indices whose outputs are concatenated into ``fused`` on every pass,
    for a drafter that reads the target's residual stream.
    """

    def __init__(self, layout: Layout, cache: list[Any], taps: Sequence[int] = ()) -> None:
        self.layout = layout
        self.cache = cache
        self.taps = tuple(taps)
        self.fused: mx.array | None = None
        self._pending: dict[int, deltanet.Pending] = {}
        self._offset = 0
        linear = [i for i in range(layout.layer_count) if layout.is_linear(i)]
        self._linear = set(linear)
        self._attention = [i for i in range(layout.layer_count) if i not in self._linear]

    def prefill(self, tokens: Sequence[int], chunk: int = 2048) -> mx.array:
        """Run ``tokens`` into the cache; returns their pre-norm hidden states [T, D]."""
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
        """Tokens already in the attention caches."""
        return int(self.cache[self._attention[0]].offset) if self._attention else 0

    def forward(self, tree: Tree) -> tuple[mx.array, mx.array]:
        """``(logits [N, V], pre-norm hidden [N, D])`` for every node of ``tree``."""
        layout = self.layout
        offset = self.offset
        self._offset = offset
        ancestors = mx.array(tree.ancestors, mx.int32)
        width = self._conv_width()
        taps = deltanet.conv_taps(tree.depth, tree.ancestors, width) if width else None
        mask = attention.tree_mask(tree.ancestors, offset)
        positions = mx.array([offset + d for d in tree.depth], mx.int32)
        hidden = layout.embed(mx.array(tree.tokens, mx.uint32))
        tapped = []
        wanted = set(self.taps)
        for index in range(layout.layer_count):
            mod, held = layout.mixer(index), self.cache[index]
            if index in self._linear:
                def mix(x, mod=mod, held=held, index=index):
                    out, self._pending[index] = deltanet.tree_mix(mod, held, x, taps, ancestors)
                    return out
            else:
                def mix(x, mod=mod, held=held):
                    return attention.attend(mod, held, x, positions, mask)
            hidden = layout.block(index, hidden, mix)
            if index in wanted:
                tapped.append(hidden)
            if index % ASYNC_EVERY == ASYNC_EVERY - 1:
                mx.async_eval(hidden)
        self.fused = mx.concatenate(tapped, axis=-1) if tapped else None
        return layout.logits(hidden), hidden

    def commit(self, path: Sequence[int]) -> None:
        """Keep the root-to-node ``path`` (node indices, starting at 0) in every cache."""
        rows = mx.array(list(path), mx.int32)
        attention.compact([self.cache[i] for i in self._attention], self._offset, rows)
        for index in self._linear:
            deltanet.commit(self.layout.mixer(index), self.cache[index],
                            self._pending.pop(index), rows)

    def _conv_width(self) -> int:
        for index in self._linear:
            return int(self.layout.mixer(index).conv_kernel_size)
        return 0
