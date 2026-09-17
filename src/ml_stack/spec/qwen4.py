"""Qwen4-Exp (Qwen3.8-Flash-Next) over a draft tree, driving mlx-vlm's ``qwen4_exp`` modules.

Every layer keeps ``hc_count`` residual streams mixed in and out of its mixer and its MoE by
hyper-connections. The PLE layer adds an n-gram embedding hashed from each node's own last
``ngram_size`` tokens and a dilated convolution along the node's path. Attention layers are
QSA: once a node's position has more complete key blocks than the indexer's budget, it
attends to the blocks its indexer selects plus the incomplete tail, scored over the blocks
that are complete along its own path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.models.qwen3_5.language import _create_qwen3_5_ssm_mask
from mlx_vlm.models.qwen4_exp.language import _create_qwen4_exp_attention_mask, _qwen4_inject

from ml_stack.spec.layout import linear_block, offset_of

if TYPE_CHECKING:
    from ml_stack.spec.verify import TreePass

__all__ = ["Qwen4ExpLayout", "attend", "tree_attend", "tree_ple"]


class _Context:
    """The two-slot view of an n-gram cache the embedding reads its prior tokens from."""

    def __init__(self, tokens: mx.array) -> None:
        self.tokens = tokens

    def __getitem__(self, index: int) -> mx.array | None:
        return self.tokens if index == 3 else None

    def update_window(self, *_: Any, **__: Any) -> None:
        return None


def tree_ple(ple: nn.Module, cache: Any, hidden: mx.array, walk: TreePass) -> mx.array:
    """The PLE layer's addition [N, hc*D] for every node, leaving its n-gram and conv commit."""
    embedding = ple.ple_embedding
    count, width = hidden.shape[0], hidden.shape[-1]
    prior = cache[3] if cache[3] is not None else mx.full(
        (1, embedding.context_len), embedding.eos_token_id, dtype=mx.int64)
    tokens = mx.concatenate([prior[0], mx.array(walk.tree.tokens, mx.int64)])
    history = tokens[walk.conv_taps(embedding.ngram_size)]
    found = embedding(history[:, -1:], _Context(history[:, :-1]))[:, 0]
    keys = ple.norm_key(ple.key_proj(found)).reshape(count, ple.hc_count, ple.hidden_size)
    values = ple.value_proj(found)
    queries = ple.norm_query(hidden).reshape(count, ple.hc_count, ple.hidden_size)
    gate = mx.sum(keys * queries, axis=-1, keepdims=True) / math.sqrt(ple.hidden_size)
    gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
    gated = (mx.sigmoid(gate) * values[:, None, :]).reshape(count, width)
    normed = ple.norm_conv(gated)
    span = ple.short_conv_state_len
    state = cache[2][0] if cache[2] is not None else mx.zeros((span, width), dtype=normed.dtype)
    rows = mx.concatenate([state, normed], axis=0)
    kernel = ple.conv1d.weight.shape[1]
    weight = ple.conv1d.weight[:, :, 0].T
    conv = nn.silu((rows[walk.conv_taps(kernel, ple.conv_dilation)] * weight).sum(axis=1))

    def commit(path: mx.array) -> None:
        context = embedding.context_len
        kept = mx.concatenate([tokens[:context], tokens[context + path]])
        cache[3] = kept[-context:][None]
        cache[2] = mx.concatenate([state, normed[path]], axis=0)[-span:][None]

    walk.commits.append(commit)
    return gated + conv


def _shared_blocks(indexer: nn.Module, cache: Any, complete: int) -> mx.array:
    """The normalised, rotated pooled keys [1, 1, B, d] of the prefix's ``complete`` blocks."""
    ratio = indexer.compress_ratio
    held = cache.index_block_keys
    have = held.shape[2] if held is not None and cache.index_block_ratio == ratio else 0
    if have >= complete:
        return held[:, :, :complete]
    raw = cache.index_keys[:, have * ratio:complete * ratio]
    pooled = mx.mean(raw.reshape(1, complete - have, ratio, -1).astype(mx.float32), axis=2)
    pooled = indexer.k_layernorm(pooled.astype(raw.dtype))[:, None]
    starts = mx.arange(have, complete, dtype=mx.int32) * ratio
    pooled = indexer._apply_rope(pooled, cache.index_position_ids[..., starts])
    blocks = mx.concatenate([held[:, :, :have], pooled], axis=2) if have else pooled
    cache.index_block_keys, cache.index_block_ratio = blocks, ratio
    return blocks


def _columns(walk: TreePass) -> list[list[int]]:
    """For each node, the key column of the token at each depth of its path."""
    return [[walk.offset + j for j in reversed([a for a in row if a >= 0])]
            for row in walk.tree.ancestors]


def _column_of(position: int, node: int, walk: TreePass, along: list[list[int]]) -> int:
    return position if position < walk.offset else along[node][position - walk.offset]


def _path_blocks(indexer: nn.Module, cache: Any, walk: TreePass, counts: list[int],
                 along: list[list[int]]) -> mx.array:
    """Pooled keys [N, 1, P, d] of the blocks that end inside the tree, along each node's path."""
    ratio = indexer.compress_ratio
    shared = walk.offset // ratio
    wide = max(counts) - shared
    table, starts = [], []
    for node, count in enumerate(counts):
        table.append([[_column_of(block * ratio + i, node, walk, along)
                       if block < count else 0 for i in range(ratio)]
                      for block in range(shared, shared + wide)])
        starts.append([block * ratio for block in range(shared, shared + wide)])
    raw = cache.index_keys[0][mx.array(table, mx.int32)]
    pooled = mx.mean(raw.astype(mx.float32), axis=2).astype(raw.dtype)
    pooled = indexer.k_layernorm(pooled)[:, None]
    return indexer._apply_rope(pooled, mx.array(starts, mx.int32))


def _selected(scores: mx.array, counts: list[int], topk: int) -> mx.array:
    """The top ``topk`` block indices [N, topk] of each node's row, over its own blocks only."""
    rows: list[mx.array | None] = [None] * len(counts)
    for count in sorted(set(counts)):
        members = [n for n, c in enumerate(counts) if c == count]
        if count <= topk:
            picked = mx.zeros((len(members), topk), dtype=mx.int32)
        else:
            group = scores[mx.array(members, mx.int32), :count]
            picked = mx.argpartition(group, kth=-topk, axis=-1)[:, -topk:].astype(mx.int32)
        for i, node in enumerate(members):
            rows[node] = picked[i]
    return mx.stack(rows)


def _sparse_mask(indexer: nn.Module, cache: Any, queries: mx.array, walk: TreePass,
                 counts: list[int]) -> mx.array:
    """The boolean mask [N, offset + N] of what each node attends to under QSA."""
    ratio, topk = indexer.compress_ratio, indexer.block_topk
    offset, count = walk.offset, len(counts)
    along = _columns(walk)
    queries = indexer._apply_rope(indexer.q_layernorm(queries)[:, :, None, :],
                                  walk.positions[:, None])
    blocks = _shared_blocks(indexer, cache, offset // ratio)
    if max(counts) > offset // ratio:
        blocks = mx.concatenate([mx.broadcast_to(blocks, (count, *blocks.shape[1:])),
                                 _path_blocks(indexer, cache, walk, counts, along)], axis=2)
    scores = queries.astype(mx.float32) @ blocks.astype(mx.float32).transpose(0, 1, 3, 2)
    scores = mx.sum(mx.maximum(scores, 0), axis=1)[:, 0] / math.sqrt(indexer.head_dim)
    picked = _selected(scores, counts, topk)
    tokens = (picked[:, :, None] * ratio + mx.arange(ratio, dtype=mx.int32)).reshape(count, -1)
    lookup = mx.array([row + [0] * (walk.tree.max_depth + 1 - len(row)) for row in along],
                      mx.int32)
    inside = mx.take_along_axis(lookup, mx.maximum(tokens - offset, 0), axis=1)
    columns = mx.where(tokens < offset, tokens, inside)
    sentinel = offset + count
    tails = [[_column_of(p, n, walk, along) for p in range(c * ratio, offset + d + 1)]
             for n, (c, d) in enumerate(zip(counts, walk.tree.depth, strict=True))]
    tail = mx.array([row + [sentinel] * (ratio - len(row)) for row in tails], mx.int32)
    columns = mx.concatenate([columns, tail], axis=1)
    chosen = mx.put_along_axis(mx.zeros((count, sentinel + 1), dtype=mx.bool_), columns,
                               mx.ones(columns.shape, dtype=mx.bool_), axis=1)[:, :sentinel]
    sparse = mx.array([c > topk for c in counts])
    return mx.where(sparse[:, None], chosen, walk.mask)


def tree_attend(mod: nn.Module, cache: Any, x: mx.array, walk: TreePass) -> mx.array:
    """QSA attention [N, D] over the tree, appending N key and indexer rows to ``cache``."""
    count = x.shape[0]
    indexer = mod.indexer
    projected = indexer.index_qk_proj(x).reshape(count, indexer.n_heads + indexer.kv_heads,
                                                 indexer.head_dim)
    cache.update_indexer(projected[None, :, indexer.n_heads], walk.positions[None])
    walk.attention.append(cache)
    counts = [(walk.offset + d + 1) // indexer.compress_ratio for d in walk.tree.depth]
    mask = walk.mask
    if max(counts) > indexer.block_topk:
        mask = _sparse_mask(indexer, cache, projected[:, :indexer.n_heads], walk, counts)
    return attend(mod, cache, x, walk.positions, mask)


def attend(mod: nn.Module, cache: Any, x: mx.array, positions: mx.array,
           mask: mx.array | str | None) -> mx.array:
    """Gated attention [N, D] for rows x [N, D] at ``positions`` [N], appending N key rows."""
    count = x.shape[0]
    rotary = mx.broadcast_to(positions.astype(mx.int32)[None, None], (3, 1, count))
    queries, keys, values, gate, _ = mod._prepare_projected_qkv(
        mod.q_proj(x)[None], mod.k_proj(x)[None], mod.v_proj(x)[None], cache, rotary, None,
        None)
    out = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=mod.scale,
                                       mask=mask)
    out = out[0].transpose(1, 0, 2).reshape(count, -1)
    return mod.o_proj(out * mx.sigmoid(gate[0]))


class Qwen4ExpLayout:
    """mlx-vlm's ``qwen4_exp``; its residual stream is ``hc_count`` copies of the hidden size."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.text = getattr(model, "language_model", model)
        self.inner = self.text.model
        self.args = self.text.args

    @property
    def layer_count(self) -> int:
        return len(self.inner.layers)

    def embed(self, tokens: mx.array) -> mx.array:
        return self.inner.embed_tokens(tokens)

    def enter(self, tokens: mx.array) -> mx.array:
        """The residual streams a pass starts from: the embedding repeated per stream."""
        return mx.tile(self.embed(tokens), (1, int(self.args.hc_count)))

    def head(self) -> nn.Module:
        """The output projection a drafter scores its own hidden states with."""
        return self.inner.embed_tokens if self.args.tie_word_embeddings else self.text.lm_head

    def contract(self, hidden: mx.array) -> mx.array:
        """The final hyper-connection mix of residual streams down to the hidden size."""
        return self.inner.hyper_connection_mixer(hidden)

    def logits(self, hidden: mx.array) -> mx.array:
        mixed = self.contract(hidden)
        if self.args.tie_word_embeddings:
            return self.inner.embed_tokens.as_linear(mixed)
        return self.text.lm_head(mixed)

    def tap(self, index: int, hidden: mx.array) -> mx.array:
        """Layer ``index``'s output as the next layer's attention hyper-connection mixes it."""
        if index + 1 >= self.layer_count:
            return self.contract(hidden)
        return self.inner.layers[index + 1].attn_hyper_connection(hidden)[0]

    def block(self, index: int, hidden: mx.array, cache: Any, walk: TreePass) -> mx.array:
        layer = self.inner.layers[index]
        if "ple" in layer:
            hidden = hidden + tree_ple(layer.ple, cache, hidden, walk)
        mixed, streams, weights = layer.attn_hyper_connection(hidden)
        if layer.is_linear:
            branch = linear_block(layer.linear_attn, cache, mixed, walk)
        else:
            branch = tree_attend(layer.self_attn, cache, mixed, walk)
        hidden = _qwen4_inject(branch, streams, weights)
        mixed, streams, weights = layer.mlp_hyper_connection(hidden)
        return _qwen4_inject(layer.mlp(mixed), streams, weights)

    def prefill(self, tokens: mx.array, cache: list[Any],
                taps: Sequence[int]) -> tuple[mx.array, list[mx.array]]:
        """Residual streams [1, T, hc*D] for ``tokens`` [1, T], and the tapped layers' outputs."""
        hidden = mx.tile(self.embed(tokens), (1, 1, int(self.args.hc_count)))
        fa_mask = _create_qwen4_exp_attention_mask(hidden, cache[self.inner.fa_idx])
        ssm_mask = _create_qwen3_5_ssm_mask(hidden, cache[self.inner.ssm_idx])
        wanted, tapped = set(taps), []
        for index, (layer, held) in enumerate(zip(self.inner.layers, cache, strict=True)):
            hidden = layer(hidden, tokens, mask=ssm_mask if layer.is_linear else fa_mask,
                           cache=held, position_ids=None)
            if index in wanted:
                tapped.append(self.tap(index, hidden))
        return hidden, tapped

    def make_cache(self) -> list[Any]:
        return self.text.make_cache()

    def offset(self, cache: list[Any]) -> int:
        return offset_of(cache)
