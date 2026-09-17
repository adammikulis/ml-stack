"""Byte-level language model over a pile of text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ml_stack.backend import detect_backend
from ml_stack.train.holdout import contiguous_tail
from ml_stack.train.recipes.built import Built
from ml_stack.train.recipes.data import VOCAB, as_bytes, lm_batches, read_text
from ml_stack.train.recipes.models import build_mlx_lm, build_torch_lm, suggest_size

CONTINUATION = 32
"""Bytes ``predict`` adds to the text it is given."""


def build_text_lm(spec: dict[str, Any], config: dict[str, Any], data: Path | None,
                  framework: str) -> Built:
    docs = read_text(data, spec.get("data", {}).get("fields", ["text"])[0]) if data else []
    if data is not None and not docs:
        raise ValueError(
            f"no text found under {data}. Expected {spec['data']['formats']} "
            f"with a {spec['data']['fields'][0]!r} field.")
    size = config.get("size") or suggest_size(spec["sizes"], None)
    shape = spec["sizes"][size]
    context = int(config["context"])
    seed = int(config.get("seed") or 0)
    framework = framework or detect_backend()

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

        def last_logits(window: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return model(torch.as_tensor(window[None]))[0, -1].numpy()
    else:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        model = build_mlx_lm(layers=shape["layers"], d_model=shape["d_model"],
                             heads=shape["heads"], context=context)
        optimizer = optim.AdamW(learning_rate=config["learning_rate"],
                                bias_correction=True)

        def loss(m, batch):
            x, y = batch
            logits = m(mx.array(x))
            return mx.mean(nn.losses.cross_entropy(
                logits.reshape(-1, VOCAB), mx.array(y).reshape(-1)))

        def last_logits(window: np.ndarray) -> np.ndarray:
            return np.array(model(mx.array(window[None]))[0, -1])

    def predict(text: Any) -> str:
        """``text`` followed by the model's greedy continuation."""
        stream = list(str(text).encode("utf-8", errors="replace")) or [0]
        for _ in range(CONTINUATION):
            window = np.array(stream[-context:], dtype=np.int64)
            stream.append(int(np.argmax(last_logits(window))))
        return bytes(stream).decode("utf-8", errors="replace")

    built = Built(model=model, optimizer=optimizer, loss=loss, predict=predict,
                  config={**config, "recipe": "text-lm", "size": size,
                          "framework": framework})
    if data is None:
        return built

    split = contiguous_tail(docs, fraction=0.05)
    train_stream = as_bytes(list(split.train))
    eval_stream = as_bytes(list(split.holdout)) if split.holdout else train_stream
    built.batches = lm_batches(train_stream, context=context,
                               batch_size=int(config["batch_size"]), seed=seed)
    built.eval_batches = lm_batches(eval_stream, context=context,
                                    batch_size=int(config["batch_size"]), seed=seed + 1)
    built.config.update({"documents": len(docs), "train_bytes": int(train_stream.size),
                         "holdout_bytes": int(eval_stream.size)})
    return built
