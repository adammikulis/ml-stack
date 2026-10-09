"""Conversations for fine-tuning: reading a dataset, rendering through a chat template, batching.

Shared by the Hugging Face recipe and the MLX adapter recipe, so neither imports the other.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

IGNORE = -100
"""The label a token gets when the loss must not see it; what torch's cross entropy skips."""

HOLDOUT_EVERY = 10


# -- reading ----------------------------------------------------------------------------------

def read_conversations(data: Path | str) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """``(train, holdout, manifest)`` from a data directory or one ``.jsonl`` file.

    ``train.jsonl`` and ``holdout.jsonl`` are taken as they are when both exist; otherwise
    every row found is split one in ten by the hash of its first user message, so a file
    that never went through the synthesiser still gets a held-out score that means
    something.
    """
    path = Path(data).expanduser()
    manifest: dict[str, Any] = {}
    if path.is_dir() and (path / "manifest.json").is_file():
        manifest = json.loads((path / "manifest.json").read_text())

    def rows_of(file: Path) -> list[dict]:
        out = []
        for line in file.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("messages"), list):
                out.append(row)
        return out

    if path.is_dir() and (path / "train.jsonl").is_file() and (path / "holdout.jsonl").is_file():
        return rows_of(path / "train.jsonl"), rows_of(path / "holdout.jsonl"), manifest

    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    rows = [r for f in files if f.is_file() for r in rows_of(f)]
    train, holdout = [], []
    for row in rows:
        user = next((m.get("content") or "" for m in row["messages"] if m.get("role") == "user"), "")
        digest = int(hashlib.sha256(str(user).strip().lower().encode()).hexdigest(), 16)
        (holdout if digest % HOLDOUT_EVERY == 0 else train).append(row)
    return train, holdout, manifest


# -- rendering ----------------------------------------------------------------------------------

def render(tokenizer: Any, messages: Sequence[Mapping[str, Any]],
           tools: Sequence[Mapping[str, Any]] | None, *, context: int
           ) -> tuple[list[int], list[int]] | None:
    """Token ids and labels for one conversation, the loss on the last assistant turn only.

    The template is rendered twice — everything before the assistant turn with the
    generation prompt, and the whole thing — and the first must be a prefix of the second;
    the tokens under that prefix are labelled ``IGNORE``. ``None`` when the context cuts
    the assistant turn off entirely, which is a row that would teach nothing.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("a conversation must end with the assistant turn to learn from")
    tools = list(tools or []) or None
    prefix = tokenizer.apply_chat_template(list(messages[:-1]), tools=tools, tokenize=False,
                                           add_generation_prompt=True)
    full = tokenizer.apply_chat_template(list(messages), tools=tools, tokenize=False)
    if not full.startswith(prefix):
        raise ValueError(
            "this chat template does not render the assistant turn as a continuation of "
            "the turns before it, so the assistant tokens cannot be found by prefix")

    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        labels = [tid if end > len(prefix) else IGNORE
                  for tid, (_, end) in zip(ids, enc["offset_mapping"], strict=True)]
    else:
        ids = list(tokenizer(full, add_special_tokens=False)["input_ids"])
        head = list(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        if ids[:len(head)] != head:
            raise ValueError("the tokenizer does not tokenise the prefix the same way on its "
                             "own; a fast tokenizer with offsets is needed here")
        labels = [IGNORE] * len(head) + ids[len(head):]

    ids, labels = ids[:context], labels[:context]
    if all(label == IGNORE for label in labels):
        return None
    return ids, labels


def conversation_batches(rendered: Sequence[tuple[list[int], list[int]]], *, batch_size: int,
                         pad_id: int, seed: int = 0):
    """``(step) -> {"input_ids", "attention_mask", "labels"}`` padded to the longest row."""
    rng = np.random.default_rng(seed)
    if not rendered:
        raise ValueError("no conversations survived rendering")

    def batch(step: int) -> dict[str, np.ndarray]:
        pick = rng.integers(0, len(rendered), size=min(batch_size, len(rendered)))
        rows = [rendered[i] for i in pick]
        width = max(len(ids) for ids, _ in rows)
        ids = np.full((len(rows), width), pad_id, dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        labels = np.full((len(rows), width), IGNORE, dtype=np.int64)
        for i, (row_ids, row_labels) in enumerate(rows):
            ids[i, :len(row_ids)] = row_ids
            mask[i, :len(row_ids)] = 1
            labels[i, :len(row_labels)] = row_labels
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return batch
