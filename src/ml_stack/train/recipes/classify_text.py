"""Byte-level classifier over labelled documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ml_stack.backend import detect_backend
from ml_stack.train.holdout import stratified
from ml_stack.train.recipes.built import Built
from ml_stack.train.recipes.data import class_batches, pad_to, read_labelled
from ml_stack.train.recipes.models import (
    build_mlx_classifier,
    build_torch_classifier,
    suggest_size,
)


def _classes(config: dict[str, Any], labels: list[str]) -> list[str]:
    """The sorted labels, or the ``classes`` a trained run recorded when there is no data."""
    classes = sorted(set(labels)) if labels else [str(c) for c in config.get("classes") or []]
    if not classes:
        raise ValueError("classify-text needs its data or the classes a trained run recorded")
    if len(classes) < 2:
        raise ValueError(f"only one label ({classes[0]!r}); there is nothing to learn")
    return classes


def build_classifier(spec: dict[str, Any], config: dict[str, Any], data: Path | None,
                     framework: str) -> Built:
    texts, labels = read_labelled(data) if data is not None else ([], [])
    if data is not None and not texts:
        raise ValueError(
            f"no labelled rows under {data}. Expected .jsonl with 'text' and 'label'.")
    classes = _classes(config, labels)

    size = config.get("size") or suggest_size(spec["sizes"], None)
    shape = spec["sizes"][size]
    context = int(config["context"])
    framework = framework or detect_backend()

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

        def probabilities(row: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return torch.softmax(model(torch.as_tensor(row[None])), -1)[0].numpy()
    else:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        model = build_mlx_classifier(
            layers=shape["layers"], d_model=shape["d_model"], heads=shape["heads"],
            context=context, classes=len(classes))
        optimizer = optim.AdamW(learning_rate=config["learning_rate"],
                                bias_correction=True)

        def loss(m, batch):
            x, y = batch
            return mx.mean(nn.losses.cross_entropy(m(mx.array(x)), mx.array(y)))

        def probabilities(row: np.ndarray) -> np.ndarray:
            return np.array(mx.softmax(model(mx.array(row[None])), axis=-1)[0])

    def predict(text: Any) -> dict[str, Any]:
        """The most likely label for ``text``, and every label's probability."""
        scores = probabilities(pad_to(str(text), context))
        return {"label": classes[int(np.argmax(scores))],
                "scores": {name: float(p) for name, p in zip(classes, scores, strict=True)}}

    built = Built(model=model, optimizer=optimizer, loss=loss, predict=predict,
                  config={**config, "recipe": "classify-text", "size": size,
                          "framework": framework, "classes": classes})
    if data is None:
        return built

    rows = list(zip(texts, labels, strict=True))
    split = stratified(rows, labels, fraction=0.2, seed=int(config.get("seed") or 0))
    train, holdout = list(split.train), list(split.holdout)
    batch_size = int(config["batch_size"])
    built.batches = class_batches([t for t, _ in train], [lab for _, lab in train], classes,
                                  context=context, batch_size=batch_size)
    built.eval_batches = class_batches([t for t, _ in holdout], [lab for _, lab in holdout],
                                       classes, context=context, batch_size=batch_size,
                                       seed=1)
    built.config.update({"rows": len(rows), "train_rows": len(train),
                         "holdout_rows": len(holdout)})
    return built
