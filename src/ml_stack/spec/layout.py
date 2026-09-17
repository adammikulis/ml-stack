"""How a hybrid model lays out its layers, for a verifier that runs them over a draft tree.

A layout names the embedding, the head and the residual stream, runs one layer over every
node of a tree with the tree mixers of `deltanet` and `attention`, and prefills a prompt
through the model's own layers, so one verifier serves every architecture that has one.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from ml_stack.spec import LAYOUTS, attention, deltanet

if TYPE_CHECKING:
    from ml_stack.spec.verify import TreePass

__all__ = ["Layout", "Qwen35Layout", "layout_for", "offset_of"]


class Layout(Protocol):
    """What the verifier and the drafters need to know about a hybrid model."""

    model: nn.Module
    args: Any

    @property
    def layer_count(self) -> int: ...

    def embed(self, tokens: mx.array) -> mx.array: ...

    def enter(self, tokens: mx.array) -> mx.array: ...

    def logits(self, hidden: mx.array) -> mx.array: ...

    def head(self) -> nn.Module: ...

    def tap(self, index: int, hidden: mx.array) -> mx.array: ...

    def block(self, index: int, hidden: mx.array, cache: Any, walk: TreePass) -> mx.array: ...

    def prefill(self, tokens: mx.array, cache: list[Any],
                taps: Sequence[int]) -> tuple[mx.array, list[mx.array]]: ...

    def make_cache(self) -> list[Any]: ...

    def offset(self, cache: list[Any]) -> int: ...


def offset_of(cache: list[Any]) -> int:
    """Tokens held by the first attention cache in ``cache``, 0 when there is none."""
    return next((int(held.offset) for held in cache if hasattr(held, "update_and_fetch")), 0)


def linear_block(mod: nn.Module, cache: Any, x: mx.array, walk: TreePass) -> mx.array:
    """A gated DeltaNet mixer over the tree, leaving its conv and state commit on ``walk``."""
    taps = walk.conv_taps(int(mod.conv_kernel_size))
    out, pending = deltanet.tree_mix(mod, cache, x, taps, walk.ancestors)
    walk.commits.append(lambda rows: deltanet.commit(mod, cache, pending, rows))
    return out


class Qwen35Layout:
    """mlx-lm's ``qwen3_5`` and ``qwen3_5_moe``: pre-norm residual blocks, dense or MoE MLP."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.text = model.language_model
        self.inner = self.text.model
        self.args = self.text.args

    @property
    def layer_count(self) -> int:
        return len(self.inner.layers)

    def embed(self, tokens: mx.array) -> mx.array:
        return self.inner.embed_tokens(tokens)

    def enter(self, tokens: mx.array) -> mx.array:
        """The residual stream a pass starts from."""
        return self.embed(tokens)

    def logits(self, hidden: mx.array) -> mx.array:
        normed = self.inner.norm(hidden)
        if self.args.tie_word_embeddings:
            return self.inner.embed_tokens.as_linear(normed)
        return self.text.lm_head(normed)

    def head(self) -> nn.Module:
        """The output projection a drafter scores its own hidden states with."""
        return self.inner.embed_tokens if self.args.tie_word_embeddings else self.text.lm_head

    def tap(self, index: int, hidden: mx.array) -> mx.array:
        """What a drafter reads of layer ``index``'s output."""
        return hidden

    def block(self, index: int, hidden: mx.array, cache: Any, walk: TreePass) -> mx.array:
        layer = self.inner.layers[index]
        x = layer.input_layernorm(hidden)
        if layer.is_linear:
            mixed = linear_block(layer.linear_attn, cache, x, walk)
        else:
            walk.attention.append(cache)
            mixed = attention.attend(layer.self_attn, cache, x, walk.positions, walk.mask)
        hidden = hidden + mixed
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    def prefill(self, tokens: mx.array, cache: list[Any],
                taps: Sequence[int]) -> tuple[mx.array, list[mx.array]]:
        """Pre-norm hidden states [1, T, D] for ``tokens`` [1, T], and the tapped layers' outputs."""
        hidden = self.inner.embed_tokens(tokens)
        fa_mask = create_attention_mask(hidden, cache[self.inner.fa_idx])
        ssm_mask = create_ssm_mask(hidden, cache[self.inner.ssm_idx])
        wanted, tapped = set(taps), []
        for index, (layer, held) in enumerate(zip(self.inner.layers, cache, strict=True)):
            hidden = layer(hidden, mask=ssm_mask if layer.is_linear else fa_mask, cache=held)
            if index in wanted:
                tapped.append(hidden)
        return hidden, tapped

    def make_cache(self) -> list[Any]:
        return self.model.make_cache()

    def offset(self, cache: list[Any]) -> int:
        return offset_of(cache)


def model_type_of(model: nn.Module) -> str:
    """The ``model_type`` of a loaded mlx-lm or mlx-vlm model."""
    kind = getattr(model, "model_type", None) or getattr(getattr(model, "config", None),
                                                          "model_type", "")
    return str(kind or "")


def layout_for(model: nn.Module) -> Layout:
    """The layout for a loaded model, by its ``model_type``."""
    kind = model_type_of(model)
    named = LAYOUTS.get(kind)
    if named is None:
        raise ValueError(f"no tree-verification layout for model_type {kind!r}; "
                         f"known: {', '.join(sorted(LAYOUTS))}")
    module, _, attribute = named.partition(":")
    return getattr(importlib.import_module(module), attribute)(model)
