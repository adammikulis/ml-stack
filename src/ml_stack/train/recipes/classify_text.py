"""Byte-level classifier over labelled documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_stack.train.holdout import stratified
from ml_stack.train.recipes import Built
from ml_stack.train.recipes.data import class_batches, read_labelled
from ml_stack.train.recipes.models import (
    build_mlx_classifier,
    build_torch_classifier,
    suggest_size,
)


def build_classifier(spec: dict[str, Any], config: dict[str, Any], data: Path,
                     framework: str) -> Built:
    texts, labels = read_labelled(data)
    if not texts:
        raise ValueError(
            f"no labelled rows under {data}. Expected .jsonl with 'text' and 'label'.")
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError(f"only one label ({classes[0]!r}); there is nothing to learn")

    rows = list(zip(texts, labels))
    split = stratified(rows, labels, fraction=0.2, seed=int(config.get("seed") or 0))
    train, holdout = list(split.train), list(split.holdout)

    size = config.get("size") or suggest_size(spec["sizes"], None)
    shape = spec["sizes"][size]
    context = int(config["context"])

    if framework == "torch":
        import torch

        model = build_torch_classifier(
            layers=shape["layers"], d_model=shape["d_model"], heads=shape["heads"],
            context=context, classes=len(classes))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

        def loss(m, batch):
            x, y = batch
            return torch.nn.functional.cross_entropy(
                m(torch.as_tensor(x)), torch.as_tensor(y))
    else:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        model = build_mlx_classifier(
            layers=shape["layers"], d_model=shape["d_model"], heads=shape["heads"],
            context=context, classes=len(classes))
        optimizer = optim.AdamW(learning_rate=config["learning_rate"])

        def loss(m, batch):
            x, y = batch
            return mx.mean(nn.losses.cross_entropy(m(mx.array(x)), mx.array(y)))

    batch_size = int(config["batch_size"])
    return Built(
        model=model, optimizer=optimizer, loss=loss,
        batches=class_batches([t for t, _ in train], [lab for _, lab in train], classes,
                              context=context, batch_size=batch_size),
        eval_batches=class_batches([t for t, _ in holdout], [lab for _, lab in holdout],
                                   classes, context=context, batch_size=batch_size,
                                   seed=1),
        config={"recipe": "classify-text", "size": size, "framework": framework,
                "classes": classes, "rows": len(rows), "train_rows": len(train),
                "holdout_rows": len(holdout), **config},
    )
