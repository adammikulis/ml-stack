"""Byte-level language model over a pile of text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_stack.train.holdout import contiguous_tail
from ml_stack.train.recipes import Built
from ml_stack.train.recipes.data import VOCAB, as_bytes, lm_batches, read_text
from ml_stack.train.recipes.models import build_mlx_lm, build_torch_lm, suggest_size


def build_text_lm(spec: dict[str, Any], config: dict[str, Any], data: Path,
                  framework: str) -> Built:
    docs = read_text(data, spec.get("data", {}).get("fields", ["text"])[0])
    if not docs:
        raise ValueError(
            f"no text found under {data}. Expected {spec['data']['formats']} "
            f"with a {spec['data']['fields'][0]!r} field.")

    split = contiguous_tail(docs, fraction=0.05)
    train_stream = as_bytes(list(split.train))
    eval_stream = as_bytes(list(split.holdout)) if split.holdout else train_stream

    size = config.get("size") or suggest_size(spec["sizes"], None)
    shape = spec["sizes"][size]
    context = int(config["context"])
    seed = int(config.get("seed") or 0)

    if framework == "torch":
        import torch

        model = build_torch_lm(layers=shape["layers"], d_model=shape["d_model"],
                               heads=shape["heads"], context=context)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

        def loss(m, batch):
            x, y = batch
            logits = m(torch.as_tensor(x))
            return torch.nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB), torch.as_tensor(y).reshape(-1))
    else:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        model = build_mlx_lm(layers=shape["layers"], d_model=shape["d_model"],
                             heads=shape["heads"], context=context)
        optimizer = optim.AdamW(learning_rate=config["learning_rate"])

        def loss(m, batch):
            x, y = batch
            logits = m(mx.array(x))
            return mx.mean(nn.losses.cross_entropy(
                logits.reshape(-1, VOCAB), mx.array(y).reshape(-1)))

    return Built(
        model=model, optimizer=optimizer, loss=loss,
        batches=lm_batches(train_stream, context=context,
                           batch_size=int(config["batch_size"]), seed=seed),
        eval_batches=lm_batches(eval_stream, context=context,
                                batch_size=int(config["batch_size"]), seed=seed + 1),
        config={"recipe": "text-lm", "size": size, "framework": framework,
                "documents": len(docs), "train_bytes": int(train_stream.size),
                "holdout_bytes": int(eval_stream.size), **config},
    )
