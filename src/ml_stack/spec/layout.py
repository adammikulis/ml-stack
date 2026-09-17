"""How a hybrid model lays out its layers, for a verifier that swaps in its own token mixers.

A layout names the embedding, the head, which layers are gated DeltaNet and which are full
attention, and how one layer wraps its mixer. The verifier supplies the mixer; the layout
runs everything around it, so two architectures share one verifier.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

__all__ = ["Layout", "Qwen35Layout", "layout_for"]

Mixer = Callable[[mx.array], mx.array]


class Layout(Protocol):
    """What the verifier needs to know about a hybrid model."""

    model: nn.Module
    args: Any

    @property
    def layer_count(self) -> int: ...

    def is_linear(self, index: int) -> bool: ...

    def mixer(self, index: int) -> nn.Module: ...

    def embed(self, tokens: mx.array) -> mx.array: ...

    def logits(self, hidden: mx.array) -> mx.array: ...

    def head(self) -> nn.Module: ...

    def block(self, index: int, hidden: mx.array, mix: Mixer) -> mx.array: ...

    def prefill(self, tokens: mx.array, cache: list[Any],
                taps: Sequence[int]) -> tuple[mx.array, list[mx.array]]: ...

    def make_cache(self) -> list[Any]: ...


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

    def is_linear(self, index: int) -> bool:
        return bool(self.inner.layers[index].is_linear)

    def mixer(self, index: int) -> nn.Module:
        layer = self.inner.layers[index]
        return layer.linear_attn if layer.is_linear else layer.self_attn

    def embed(self, tokens: mx.array) -> mx.array:
        return self.inner.embed_tokens(tokens)

    def logits(self, hidden: mx.array) -> mx.array:
        normed = self.inner.norm(hidden)
        if self.args.tie_word_embeddings:
            return self.inner.embed_tokens.as_linear(normed)
        return self.text.lm_head(normed)

    def head(self) -> nn.Module:
        """The output projection a drafter scores its own hidden states with."""
        return self.inner.embed_tokens if self.args.tie_word_embeddings else self.text.lm_head

    def block(self, index: int, hidden: mx.array, mix: Mixer) -> mx.array:
        layer = self.inner.layers[index]
        hidden = hidden + mix(layer.input_layernorm(hidden))
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


LAYOUTS: dict[str, type[Qwen35Layout]] = {
    "qwen3_5": Qwen35Layout,
    "qwen3_5_moe": Qwen35Layout,
}


def layout_for(model: nn.Module) -> Layout:
    """The layout for a loaded mlx-lm model, by its ``model_type``."""
    kind = str(getattr(model, "model_type", "") or "")
    made = LAYOUTS.get(kind)
    if made is None:
        raise ValueError(f"no tree-verification layout for model_type {kind!r}; "
                         f"known: {', '.join(sorted(LAYOUTS))}")
    return made(model)
