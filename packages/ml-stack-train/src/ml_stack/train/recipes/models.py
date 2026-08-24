"""A small transformer, in torch and in mlx."""

from __future__ import annotations

from typing import Any

from ml_stack.train.recipes.data import VOCAB


def build_torch_lm(*, layers: int, d_model: int, heads: int, context: int,
                   vocab: int = VOCAB) -> Any:
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                     nn.Linear(4 * d_model, d_model))

        def forward(self, x, mask):
            h = self.norm1(x)
            x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
            return x + self.mlp(self.norm2(x))

    class LM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok = nn.Embedding(vocab, d_model)
            self.pos = nn.Embedding(context, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)
            self.context = context

        def forward(self, idx):
            n = idx.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf"), device=idx.device), 1)
            pos = torch.arange(n, device=idx.device)
            x = self.tok(idx) + self.pos(pos)[None]
            for block in self.blocks:
                x = block(x, mask)
            return self.head(self.norm(x))

    return LM()


def build_mlx_lm(*, layers: int, d_model: int, heads: int, context: int,
                 vocab: int = VOCAB) -> Any:
    import mlx.core as mx
    import mlx.nn as nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiHeadAttention(d_model, heads)
            self.norm2 = nn.LayerNorm(d_model)
            self.fc1 = nn.Linear(d_model, 4 * d_model)
            self.fc2 = nn.Linear(4 * d_model, d_model)

        def __call__(self, x, mask):
            h = self.norm1(x)
            x = x + self.attn(h, h, h, mask)
            h = self.norm2(x)
            return x + self.fc2(nn.gelu(self.fc1(h)))

    class LM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok = nn.Embedding(vocab, d_model)
            self.pos = nn.Embedding(context, d_model)
            self.blocks = [Block() for _ in range(layers)]
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)
            self.context = context

        def __call__(self, idx):
            n = idx.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(n)
            x = self.tok(idx) + self.pos(mx.arange(n))[None]
            for block in self.blocks:
                x = block(x, mask)
            return self.head(self.norm(x))

    return LM()


def build_torch_classifier(*, layers: int, d_model: int, heads: int, context: int,
                           classes: int, vocab: int = VOCAB) -> Any:
    import torch.nn as nn

    body = build_torch_lm(layers=layers, d_model=d_model, heads=heads, context=context,
                          vocab=vocab)

    class Classifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = body
            self.body.head = nn.Identity()
            self.out = nn.Linear(d_model, classes)

        def forward(self, idx):
            return self.out(self.body(idx).mean(dim=1))

    return Classifier()


def build_mlx_classifier(*, layers: int, d_model: int, heads: int, context: int,
                         classes: int, vocab: int = VOCAB) -> Any:
    import mlx.core as mx
    import mlx.nn as nn

    body = build_mlx_lm(layers=layers, d_model=d_model, heads=heads, context=context,
                        vocab=vocab)

    class Classifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = body
            self.body.head = nn.Identity()
            self.out = nn.Linear(d_model, classes)

        def __call__(self, idx):
            return self.out(mx.mean(self.body(idx), axis=1))

    return Classifier()


def parameter_count(model: Any) -> int:
    try:
        return sum(p.numel() for p in model.parameters())
    except (AttributeError, TypeError):
        pass
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def suggest_size(sizes: dict[str, dict], memory_gb: float | None) -> str:
    """The biggest size that fits, or the smallest if nothing does."""
    if not sizes:
        return ""
    ordered = sorted(sizes.items(), key=lambda kv: kv[1].get("needs_gb", 0))
    if memory_gb is None:
        return ordered[0][0]
    fits = [name for name, s in ordered if s.get("needs_gb", 0) <= memory_gb]
    return fits[-1] if fits else ordered[0][0]
