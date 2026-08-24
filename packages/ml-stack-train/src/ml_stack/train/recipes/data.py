"""Reading a dataset directory into batches, without a tokenizer file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VOCAB = 256


def read_text(path: Path | str, field: str = "text") -> list[str]:
    """Every document under `path`. Accepts .jsonl, .txt, or a directory of either."""
    path = Path(path).expanduser()
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    out: list[str] = []
    for f in files:
        if f.suffix == ".jsonl":
            for line in f.read_text(errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = row.get(field) if isinstance(row, dict) else row
                if isinstance(value, str) and value.strip():
                    out.append(value)
        elif f.suffix in (".txt", ".md"):
            text = f.read_text(errors="replace")
            if text.strip():
                out.append(text)
    return out


def read_labelled(path: Path | str, text_field: str = "text",
                  label_field: str = "label") -> tuple[list[str], list[str]]:
    """Documents and their labels from .jsonl."""
    path = Path(path).expanduser()
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    texts, labels = [], []
    for f in files:
        for line in f.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            text, label = row.get(text_field), row.get(label_field)
            if isinstance(text, str) and text.strip() and label is not None:
                texts.append(text)
                labels.append(str(label))
    return texts, labels


def as_bytes(docs: list[str]) -> np.ndarray:
    """One flat uint8 stream. Byte-level, so any text works with no vocab file."""
    joined = "\n\n".join(docs)
    return np.frombuffer(joined.encode("utf-8", errors="replace"), dtype=np.uint8)


def lm_batches(stream: np.ndarray, *, context: int, batch_size: int, seed: int = 0):
    """(step) -> (inputs, targets) of shape (batch, context), targets shifted by one."""
    rng = np.random.default_rng(seed)
    if len(stream) <= context + 1:
        raise ValueError(
            f"only {len(stream)} bytes of text; needs more than {context + 1}")

    def batch(step: int):
        starts = rng.integers(0, len(stream) - context - 1, size=batch_size)
        x = np.stack([stream[s:s + context] for s in starts]).astype(np.int64)
        y = np.stack([stream[s + 1:s + context + 1] for s in starts]).astype(np.int64)
        return x, y

    return batch


def pad_to(text: str, context: int) -> np.ndarray:
    raw = text.encode("utf-8", errors="replace")[:context]
    out = np.zeros(context, dtype=np.int64)
    out[:len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return out


def class_batches(texts: list[str], labels: list[str], classes: list[str], *,
                  context: int, batch_size: int, seed: int = 0):
    """(step) -> (inputs, class indices)."""
    rng = np.random.default_rng(seed)
    index = {name: i for i, name in enumerate(classes)}
    encoded = np.stack([pad_to(t, context) for t in texts])
    targets = np.array([index[label] for label in labels], dtype=np.int64)

    def batch(step: int):
        pick = rng.integers(0, len(texts), size=min(batch_size, len(texts)))
        return encoded[pick], targets[pick]

    return batch
