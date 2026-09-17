"""Gated full attention over a draft tree: each node sees the prefix and its own ancestors."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

__all__ = ["attend", "compact", "rewind", "tree_mask"]


def tree_mask(ancestors: Sequence[Sequence[int]], offset: int) -> mx.array:
    """Boolean [N, offset + N]: the cached prefix, then node j visible to node i iff j is on i's path."""
    count = len(ancestors)
    own = [[False] * count for _ in range(count)]
    for i, row in enumerate(ancestors):
        for j in row:
            if j >= 0:
                own[i][j] = True
    return mx.concatenate([mx.ones((count, offset), dtype=mx.bool_), mx.array(own)], axis=1)


def attend(mod: nn.Module, cache: object, x: mx.array, positions: mx.array,
           mask: mx.array | None) -> mx.array:
    """The mixer output [N, D] for x [N, D] at per-node ``positions``, appending N rows to ``cache``."""
    count = x.shape[0]
    queries, gate = mx.split(mod.q_proj(x).reshape(count, mod.num_attention_heads, -1), 2,
                             axis=-1)
    keys = mod.k_norm(mod.k_proj(x).reshape(count, mod.num_key_value_heads, -1))
    values = mod.v_proj(x).reshape(count, mod.num_key_value_heads, -1)
    # one batch element per node, so each gets its own rotary position
    queries = mod.rope(mod.q_norm(queries)[:, :, None, :], offset=positions)
    keys = mod.rope(keys[:, :, None, :], offset=positions)
    queries = queries[:, :, 0, :].transpose(1, 0, 2)[None]
    keys = keys[:, :, 0, :].transpose(1, 0, 2)[None]
    values = values.transpose(1, 0, 2)[None]
    keys, values = cache.update_and_fetch(keys, values)
    out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=mod.scale, mask=mask)
    out = out[0].transpose(1, 0, 2).reshape(count, -1)
    return mod.o_proj(out * mx.sigmoid(gate.reshape(count, -1)))


def _index_rows(cache: object) -> tuple[mx.array, mx.array] | None:
    keys = getattr(cache, "index_keys", None)
    return None if keys is None else (keys, cache.index_position_ids)


def compact(caches: Sequence[object], offset: int, path: mx.array) -> None:
    """Move the accepted rows of every attention cache to ``offset`` and cut the rest.

    A cache that also keeps indexer keys (``index_keys`` [B, L, D] and their positions) has
    those rows moved the same way.
    """
    rows = [(c.keys[..., offset + path, :], c.values[..., offset + path, :]) for c in caches]
    indexed = [_index_rows(c) for c in caches]
    kept = [None if held is None else
            (mx.concatenate([held[0][:, :offset], held[0][:, offset + path]], axis=1),
             mx.concatenate([held[1][..., :offset], held[1][..., offset + path]], axis=-1))
            for held in indexed]
    mx.eval(*[half for pair in rows for half in pair],
            *[half for pair in kept if pair is not None for half in pair])
    accepted = int(path.size)
    for cache, (keys, values), index in zip(caches, rows, kept, strict=True):
        cache.keys[..., offset:offset + accepted, :] = keys
        cache.values[..., offset:offset + accepted, :] = values
        cache.offset = offset + accepted
        if index is not None:
            cache.index_keys, cache.index_position_ids = index


def rewind(cache: object, count: int) -> None:
    """Cut an attention cache back to its first ``count`` tokens, indexer keys and blocks included."""
    cache.offset = count
    if getattr(cache, "index_keys", None) is not None:
        cache.index_keys = cache.index_keys[:, :count]
        cache.index_position_ids = cache.index_position_ids[..., :count]
    blocks = getattr(cache, "index_block_keys", None)
    if blocks is not None:
        cache.index_block_keys = blocks[:, :, :count // int(cache.index_block_ratio)]
