"""The DFlash block drafter: one pass drafts a whole block, and its candidate lattice becomes a tree.

The drafter reads the target's residual stream after ``target_layer_ids`` as context rows and
fills ``block_size - 1`` mask slots after the root in one forward. DFlash2's selector scores
transitions between adjacent slots, which gives each drafted token its children for free.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ml_stack import home
from ml_stack.spec.drafters import Budget
from ml_stack.spec.drafters.dflash_model import DFlashConfig, DFlashDraftModel
from ml_stack.spec.drafters.vocab import sub_head
from ml_stack.spec.layout import Layout
from ml_stack.spec.tree import Candidates, Tree, budgeted

__all__ = ["CALIBRATIONS", "DFlashDrafter", "calibration_for", "load_dflash"]

#: measured acceptance per drafted value and depth, one file per drafter, named after it
CALIBRATIONS = Path(__file__).resolve().parent.parent / "data" / "calibration"

Calibration = Callable[[list[float], list[int]], list[float]]


def _config(path: Path) -> DFlashConfig:
    raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
    extra = raw.get("dflash_config", {})
    rope = raw.get("rope_parameters") or {}
    return DFlashConfig(
        hidden_size=raw["hidden_size"], num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"], head_dim=raw["head_dim"],
        intermediate_size=raw["intermediate_size"], vocab_size=raw["vocab_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw.get("rope_theta", rope.get("rope_theta", 1e6)),
        max_position_embeddings=raw["max_position_embeddings"],
        block_size=int(extra.get("block_size", raw.get("block_size"))),
        target_layer_ids=tuple(extra.get("target_layer_ids") or raw["target_layer_ids"]),
        mask_token_id=extra.get("mask_token_id", raw.get("mask_token_id", 0)),
        rope_scaling=raw.get("rope_scaling"),
        layer_types=tuple(raw.get("layer_types")
                          or ["full_attention"] * raw["num_hidden_layers"]),
        sliding_window=raw.get("sliding_window") if raw.get("use_sliding_window", True) else None,
        final_logit_softcapping=extra.get("final_logit_softcapping",
                                          raw.get("final_logit_softcapping")),
        selector_rank=int(extra.get("selector_rank") or 0),
        selector_top_k=int(extra.get("selector_top_k") or 0),
        conv_kernel_size=int(extra.get("conv_kernel_size") or 0),
        conv_group_size=int(extra.get("conv_group_size") or 16),
        output_multiplier=float(extra.get("output_multiplier") or 1.0))


def load_dflash(path: Path, bits: int = 4) -> tuple[DFlashDraftModel, DFlashConfig]:
    """The drafter, quantized to ``bits``; the quantized weights are cached on first load."""
    config = _config(path)
    model = DFlashDraftModel(config)

    def linear(name: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear) and "_conv" not in name \
            and "candidate_selector" not in name

    cached = home.cache("spec", "dflash", f"{path.name}-{bits}bit.safetensors")
    if bits and cached.is_file():
        nn.quantize(model, group_size=64, bits=bits, class_predicate=linear)
        model.load_weights(list(mx.load(str(cached)).items()))
    else:
        weights = {}
        for shard in sorted(path.glob("*.safetensors")):
            weights.update(mx.load(str(shard)))
        model.load_weights(list(weights.items()))
        if bits:
            nn.quantize(model, group_size=64, bits=bits, class_predicate=linear)
            mx.eval(model.parameters())
            cached.parent.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(cached), dict(tree_flatten(model.parameters())))
    mx.eval(model.parameters())
    return model, config


def calibration_for(name: str) -> Calibration | None:
    """Map a drafted value to its measured acceptance, when a calibration ships for ``name``."""
    found = CALIBRATIONS / f"{name}.json"
    if not found.is_file():
        return None
    groups = json.loads(found.read_text(encoding="utf-8"))
    by_depth = {d: (np.array(g["log10_v"]), np.array(g["p"]))
                for g in groups.values() for d in g["depths"]}
    deepest = max(by_depth)

    def calibrated(values: list[float], depths: list[int]) -> list[float]:
        raw = np.asarray(values, np.float64)
        logged = np.log10(np.clip(raw, 1e-6, 1.0))
        capped = np.minimum(np.asarray(depths), deepest)
        out = raw.copy()
        for depth in np.unique(capped):
            if int(depth) == 0:
                continue
            xs, ps = by_depth[int(depth)]
            rows = capped == depth
            out[rows] = np.interp(logged[rows], xs, ps)
        out[0] = 1.0
        return out.tolist()

    return calibrated


class ContextCache:
    """K/V rows of the drafter's context: the last ``window`` rows plus scratch for one block."""

    GROW = 512

    def __init__(self, window: int, scratch: int) -> None:
        self.window, self.scratch = window, scratch
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = self.start = self.end = 0

    def _compact(self, incoming: int, like: mx.array) -> tuple[mx.array, mx.array]:
        keep = 0 if self.keys is None else min(self.window, self.end - self.start)
        size = keep + incoming + self.scratch + self.GROW
        batch, heads, _, dim = like.shape
        keys = mx.zeros((batch, heads, size, dim), dtype=like.dtype)
        values = mx.zeros((batch, heads, size, dim), dtype=like.dtype)
        if keep and self.keys is not None and self.values is not None:
            keys[..., :keep, :] = self.keys[..., self.end - keep:self.end, :]
            values[..., :keep, :] = self.values[..., self.end - keep:self.end, :]
        self.keys, self.values, self.start, self.end = keys, values, 0, keep
        return keys, values

    def append(self, keys: mx.array, values: mx.array) -> None:
        count = keys.shape[2]
        held_keys, held_values = self.keys, self.values
        if held_keys is None or held_values is None \
                or self.end + count + self.scratch > held_keys.shape[2]:
            held_keys, held_values = self._compact(count, keys)
        held_keys[..., self.end:self.end + count, :] = keys
        held_values[..., self.end:self.end + count, :] = values
        self.end += count
        self.offset += count
        self.start = max(self.start, self.end - self.window)

    def with_block(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """The context rows followed by the block's own, without appending the block."""
        held_keys, held_values = self.keys, self.values
        if held_keys is None or held_values is None:
            return keys, values
        count = keys.shape[2]
        held_keys[..., self.end:self.end + count, :] = keys
        held_values[..., self.end:self.end + count, :] = values
        return (held_keys[..., self.start:self.end + count, :],
                held_values[..., self.start:self.end + count, :])

    def state(self) -> object:
        if self.keys is None or self.values is None:
            return None
        keys = mx.contiguous(self.keys[..., self.start:self.end, :])
        values = mx.contiguous(self.values[..., self.start:self.end, :])
        mx.eval(keys, values)
        return self.offset, keys, values

    def restore(self, state: object) -> None:
        self.keys = self.values = None
        self.offset = self.start = self.end = 0
        if state is not None:
            offset, keys, values = state  # type: ignore[misc]
            self.append(keys, values)
            self.offset = offset


class DFlashDrafter:
    """``branch`` children per drafted node, ``frontier`` nodes carried per slot."""

    def __init__(self, layout: Layout, model: DFlashDraftModel, config: DFlashConfig,
                 budget: Budget, *, calibration: Calibration | None = None) -> None:
        self.layout, self.model, self.config, self.budget = layout, model, config, budget
        self.block = max(2, min(config.block_size, budget.max_nodes))
        self.taps = tuple(config.target_layer_ids)
        self.vocab = sub_head(layout.head(), int(config.vocab_size))
        self.window = (config.sliding_window or 10**9) - self.block
        self.calibration = calibration
        self.branch, self.frontier, self.min_value, self.draft_cost = 6, 12, 0.02, 0.08
        self.caches = self._fresh()

    def _fresh(self) -> list[ContextCache]:
        return [ContextCache(self.window, self.block) for _ in self.model.layers]

    def _append(self, fused: mx.array) -> None:
        context = self.model.project_ctx(fused[None])
        count = context.shape[1]
        for layer, cache in zip(self.model.layers, self.caches, strict=True):
            attn, rows = layer.self_attn, context
            if count > self.window:
                cache.offset += count - self.window
                rows = context[:, -self.window:]
            width = rows.shape[1]
            keys = attn.k_norm(attn.k_proj(rows).reshape(1, width, attn.n_kv_heads, -1))
            values = attn.v_proj(rows).reshape(1, width, attn.n_kv_heads, -1)
            cache.append(self.model.rope(keys.transpose(0, 2, 1, 3), offset=cache.offset),
                         values.transpose(0, 2, 1, 3))

    def prefill(self, tokens: Sequence[int], hidden: mx.array) -> None:
        self.caches = self._fresh()
        self._append(hidden)

    def accept(self, tokens: Sequence[int], hidden: mx.array) -> None:
        self._append(hidden)

    def state(self) -> object:
        return [cache.state() for cache in self.caches]

    def restore(self, state: object) -> None:
        for cache, held in zip(self.caches, state, strict=True):  # type: ignore[call-overload]
            cache.restore(held)

    def _attend(self, attn: nn.Module, x: mx.array, cache: ContextCache) -> mx.array:
        width = x.shape[1]
        rope = self.model.rope
        queries = attn.q_norm(attn.q_proj(x).reshape(1, width, attn.n_heads, -1))
        keys = attn.k_norm(attn.k_proj(x).reshape(1, width, attn.n_kv_heads, -1))
        values = attn.v_proj(x).reshape(1, width, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = rope(queries.transpose(0, 2, 1, 3), offset=cache.offset)
        keys = rope(keys.transpose(0, 2, 1, 3), offset=cache.offset)
        keys, values = cache.with_block(keys, values)
        out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=attn.scale,
                                                   mask="causal")
        return attn.o_proj(out.transpose(0, 2, 1, 3).reshape(1, width, -1))

    def _hidden(self, block: mx.array) -> mx.array:
        hidden = self.layout.embed(block)
        for layer, cache in zip(self.model.layers, self.caches, strict=True):
            x = layer.input_layernorm(hidden)
            if layer.attention_conv is not None:
                x, after = layer.attention_conv.prepare(x)
                hidden = hidden + layer.attention_conv.finish(self._attend(layer.self_attn, x,
                                                                            cache), after)
            else:
                hidden = hidden + self._attend(layer.self_attn, x, cache)
            x = layer.post_attention_layernorm(hidden)
            if layer.mlp_conv is not None:
                x, after = layer.mlp_conv.prepare(x)
                hidden = hidden + layer.mlp_conv.finish(layer.mlp(x), after)
            else:
                hidden = hidden + layer.mlp(x)
        return self.model.norm(hidden[:, 1:])[0]

    def draft(self, root: int, hidden: mx.array) -> Tree:
        selector = self.model.candidate_selector
        if selector is None:
            raise ValueError("this drafter has no candidate selector; a tree needs DFlash2")
        block = mx.array([[root] + [self.config.mask_token_id] * (self.block - 1)], mx.uint32)
        states = self._hidden(block)
        logits = self.vocab(states) if self.vocab is not None else self.layout.head()(states)
        k = selector.top_k
        rows = mx.argpartition(logits, kth=-k, axis=-1)[:, -k:]
        unary = self.model._transform_unary(mx.take_along_axis(logits, rows, axis=-1))
        ids = (self.vocab.token(rows) if self.vocab is not None else rows).astype(mx.int32)
        probs = mx.softmax(selector.lattice(ids, unary, states, root), axis=-1)
        mx.eval(probs, ids)
        return self._tree(root, probs.tolist(), ids.tolist(), k)

    def _tree(self, root: int, probs: list, ids: list, k: int) -> Tree:
        tokens, parent, values = [root], [-1], [1.0]
        frontier = [(0, 0, 1.0)]
        for slot in range(len(ids)):
            grown = []
            for node, before, value in frontier:
                row = probs[slot][before]
                seen: set[int] = set()
                for i in sorted(range(k), key=lambda i: -row[i])[:self.branch]:
                    held, token = value * row[i], ids[slot][i]
                    if held < self.min_value or token in seen:
                        continue
                    seen.add(token)
                    tokens.append(token)
                    parent.append(node)
                    values.append(held)
                    grown.append((len(tokens) - 1, i, held))
            frontier = sorted(grown, key=lambda g: -g[2])[:self.frontier]
            if not frontier:
                break
        found = Candidates(tokens, parent, values)
        if self.calibration is not None:
            depth = Tree(tokens, parent).depth
            found = Candidates(tokens, parent, self.calibration(values, depth))
        return budgeted(found, self.budget.cost, draft_cost=self.draft_cost,
                        max_nodes=self.budget.max_nodes)
