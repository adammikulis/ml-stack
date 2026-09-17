"""The target's own multi-token-prediction head, expanded by beam search into a draft tree.

One step of the head reads a pair (token t, hidden h) through one full-attention decoder
layer. The pair (x_{i+1}, h_i) predicts x_{i+2}, and a drafted child reads its parent's output
as its hidden state; the head's ``readout`` of an output is what the target's head scores.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer

from ml_stack.spec.attention import attend
from ml_stack.spec.drafters import Budget
from ml_stack.spec.drafters.vocab import sub_head
from ml_stack.spec.layout import Layout
from ml_stack.spec.tree import Candidates, Tree, budgeted

__all__ = ["Beam", "Head", "MtpDrafter", "MtpHead", "load_head"]

NORMS = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight",
         ".k_norm.weight", "norm.weight", "pre_fc_norm_embedding.weight",
         "pre_fc_norm_hidden.weight")


class Head(Protocol):
    """One step of a multi-token-prediction head, and what its output is scored through."""

    def step(self, layout: Layout, cache: KVCache, pair: tuple[mx.array, mx.array],
             positions: mx.array, mask: mx.array | str | None) -> mx.array: ...

    def readout(self, out: mx.array) -> mx.array: ...


class MtpHead(nn.Module):
    """Qwen3.5's head: one pre-norm decoder layer over ``fc`` of the normed pair."""

    def __init__(self, args: object) -> None:
        super().__init__()
        width = args.hidden_size
        self.fc = nn.Linear(2 * width, width, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(width, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(width, eps=args.rms_norm_eps)
        self.layers = [DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]
        self.norm = nn.RMSNorm(width, eps=args.rms_norm_eps)

    def step(self, layout: Layout, cache: KVCache, pair: tuple[mx.array, mx.array],
             positions: mx.array, mask: mx.array | str | None) -> mx.array:
        """The head's output [W, D] for (token, hidden) pairs at ``positions``."""
        tokens, hidden = pair
        layer = self.layers[0]
        x = mx.concatenate([self.pre_fc_norm_embedding(layout.embed(tokens)),
                            self.pre_fc_norm_hidden(hidden)], axis=-1)
        x = self.fc(x)
        x = x + attend(layer.self_attn, cache, layer.input_layernorm(x), positions, mask)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.norm(x)

    def readout(self, out: mx.array) -> mx.array:
        return out


def load_head(path: Path, args: object, bits: int = 4) -> MtpHead:
    """The head's weights from a directory of safetensors, quantized to ``bits`` unless stored so."""
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    head = MtpHead(args)
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    clean = {}
    for key, value in weights.items():
        if key.startswith("mtp."):
            key = key[len("mtp."):]
            if value.ndim == 1 and key.endswith(NORMS):
                value = value + 1.0
        clean[key] = value
    stored = config.get("quantization")
    if stored:
        nn.quantize(head, group_size=stored["group_size"], bits=stored["bits"],
                    class_predicate=lambda p, m: isinstance(m, nn.Linear) and f"{p}.scales" in clean)
    head.load_weights(list(clean.items()), strict=True)
    if bits and not stored:
        nn.quantize(head, group_size=64, bits=bits)
    mx.eval(head.parameters())
    return head


@dataclass(frozen=True)
class Beam:
    """``width`` paths kept per level, ``top_k`` children each, ``depth`` levels."""

    depth: int = 6
    width: int = 6
    top_k: int = 6


BEAM = Beam()


class MtpDrafter:
    """Drafts by beam search over the head's own predictions."""

    taps: tuple[int, ...] = ()

    def __init__(self, layout: Layout, head: Head, budget: Budget, beam: Beam = BEAM) -> None:
        self.layout, self.head, self.budget = layout, head, budget
        self.depth, self.beam, self.top_k = beam.depth, beam.width, beam.top_k
        self.vocab = sub_head(layout.head(), int(layout.args.vocab_size))
        self.cache = KVCache()
        self.pairs = 0
        self.last = mx.zeros((0,))
        self.min_value, self.cost_per_level = 0.02, 0.04

    def _scores(self, out: mx.array) -> mx.array:
        read = self.head.readout(out)
        logits = self.vocab(read) if self.vocab is not None else self.layout.head()(read)
        logits = logits.astype(mx.float32)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def prefill(self, tokens: Sequence[int], hidden: mx.array) -> None:
        count = hidden.shape[0]
        self.cache, self.pairs = KVCache(), 0
        if count > 1:
            self.head.step(self.layout, self.cache,
                           (mx.array(list(tokens[1:count]), mx.uint32), hidden[:count - 1]),
                           mx.arange(count - 1), "causal")
            self.pairs = count - 1
        self.last = hidden[count - 1]

    def accept(self, tokens: Sequence[int], hidden: mx.array) -> None:
        self.cache.offset = self.pairs
        count = len(tokens)
        paired = mx.concatenate([self.last[None], hidden[:count - 1]], axis=0)
        self.head.step(self.layout, self.cache, (mx.array(list(tokens), mx.uint32), paired),
                       mx.arange(self.pairs, self.pairs + count), "causal" if count > 1 else None)
        self.pairs += count
        self.last = hidden[count - 1]

    def state(self) -> object:
        return self.cache, self.pairs, self.last

    def restore(self, state: object) -> None:
        self.cache, self.pairs, self.last = state  # type: ignore[misc]
        self.cache.offset = self.pairs

    def draft(self, root: int, hidden: mx.array) -> Tree:
        self.cache.offset = self.pairs
        k = self.top_k
        tokens = mx.array([root], mx.uint32)
        states = self.last[None]
        total = mx.zeros((1,), dtype=mx.float32)
        lineage = mx.zeros((1, 0), dtype=mx.int32)
        drafted, levels, chosen = 0, [], []
        for level in range(self.depth):
            width = tokens.shape[0]
            own = mx.arange(drafted, drafted + width)[:, None]
            reach = mx.concatenate([lineage, own], axis=1)
            seen = mx.any(reach[:, :, None] == mx.arange(drafted + width)[None, None, :], axis=1)
            mask = mx.concatenate([mx.ones((width, self.pairs), dtype=mx.bool_), seen], axis=1)
            out = self.head.step(self.layout, self.cache, (tokens, states),
                                 mx.full((width,), self.pairs + level), mask)
            scores = self._scores(out)
            top = mx.argpartition(-scores, k - 1, axis=-1)[:, :k]
            top_scores = mx.take_along_axis(scores, top, axis=-1)
            order = mx.argsort(-top_scores, axis=-1)
            top = mx.take_along_axis(top, order, axis=-1)
            top_scores = mx.take_along_axis(top_scores, order, axis=-1)
            ids = self.vocab.token(top) if self.vocab is not None else top
            summed = total[:, None] + top_scores
            levels.append((ids, summed))
            if level == self.depth - 1:
                break
            flat = summed.reshape(-1)
            picked = mx.argsort(-flat)[:min(self.beam, width * k)]
            chosen.append(picked)
            rows = picked // k
            tokens = ids.reshape(-1)[picked].astype(mx.uint32)
            states, total = out[rows], flat[picked]
            lineage = mx.concatenate([lineage[rows], own[rows]], axis=1)
            drafted += width
        mx.eval(*[t for pair in levels for t in pair], *chosen)
        return self._tree(root, levels, chosen)

    def _tree(self, root: int, levels: list, chosen: list) -> Tree:
        tokens, parent, values, starts = [root], [-1], [1.0], []
        for level, (ids, summed) in enumerate(levels):
            ids, summed = ids.tolist(), summed.tolist()
            picked = chosen[level - 1].tolist() if level else None
            starts.append(len(tokens))
            for row in range(len(ids)):
                above = 0 if picked is None else starts[level - 1] + picked[row]
                for i in range(self.top_k):
                    tokens.append(ids[row][i])
                    parent.append(above)
                    values.append(math.exp(summed[row][i]))
        seen = set()
        for i in range(1, len(tokens)):
            key = (parent[i], tokens[i])
            if key in seen or values[i] < self.min_value:
                values[i] = 0.0
            seen.add(key)
        return budgeted(Candidates(tokens, parent, values), self.budget.cost,
                        draft_cost=self.depth * self.cost_per_level,
                        max_nodes=self.budget.max_nodes)
