"""Fine-tune a chat model to call a project's tools, on conversations made for it.

The data is what ``ml-stack-train-tools`` synthesises — or anything in the same shape: rows
of ``{"messages": [...], "tools": [...]}`` — and the base is a Hugging Face causal LM with
a chat template, ``google/functiongemma-270m-it`` unless the data's manifest names another.
Each conversation is rendered through the model's own template, so what it learns is the
exact format it will be served with, and the loss is on the assistant turn only: the
question and the tool declarations are read, never predicted.

This is torch whatever the machine's default backend is, because the checkpoint decides:
a Hugging Face safetensors model loads through ``transformers``, which is torch. The
device is the machine's accelerator unless ``ML_STACK_DEVICE`` says otherwise (``cpu`` in
the tests, so a tiny model never touches a GPU that a benchmark may be using).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ml_stack.train.recipes import Built

IGNORE = -100
"""The label a token gets when the loss must not see it; what torch's cross entropy skips."""

HOLDOUT_EVERY = 10


def device_for() -> Any:
    """The torch device a fine-tune runs on: ``ML_STACK_DEVICE`` if set, else the accelerator.

    A string naming the gap when torch is absent, so a ``--dry-run`` can still print its
    plan on a machine that has yet to install it.
    """
    from ml_stack.backend.device import resolve_torch_device
    try:
        return resolve_torch_device(os.environ.get("ML_STACK_DEVICE") or None)
    except ImportError:
        return "torch (not installed)"


# -- reading ----------------------------------------------------------------------------------

def read_conversations(data: Path | str) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """``(train, holdout, manifest)`` from a data directory or one ``.jsonl`` file.

    ``train.jsonl`` and ``holdout.jsonl`` are taken as they are when both exist; otherwise
    every row found is split one in ten by the hash of its first user message, so a file
    that never went through the synthesiser still gets a held-out score that means
    something.
    """
    import hashlib

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
                  for tid, (_, end) in zip(ids, enc["offset_mapping"])]
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


# -- the model ------------------------------------------------------------------------------------

def load_base(base: str, *, device: Any = None) -> tuple[Any, Any]:
    """``(model, tokenizer)`` for a Hugging Face causal LM, the model wrapped for checkpointing.

    Whatever ``transformers`` cannot import is reported as the seam it is, rather than as
    a missing attribute three frames down.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ValueError(
            "the tool-calls recipe fine-tunes a Hugging Face checkpoint and needs torch and "
            "transformers: pip install 'ml-stack[torch]' transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(base)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{base} has no chat template, so there is no format to teach; "
                         "pick an instruction-tuned base")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model.config.use_cache = False
    model.to(device or device_for())
    return checkpointable(model), tokenizer


_CHECKPOINTABLE: Any = None


def checkpointable(lm: Any) -> Any:
    """A Hugging Face causal LM wrapped so its state dict names each tensor once.

    Measured on a Gemma3 model: ``state_dict()`` holds ``lm_head.weight`` and
    ``model.embed_tokens.weight`` as one storage under two names, and safetensors refuses
    to write shared tensors — so the existing ``TorchStep`` would fail at its first
    checkpoint. Here the duplicate is dropped on the way out and re-tied on the way back
    in, which is what ``save_pretrained`` does too, and the trainer is used unchanged.
    The wrapper's ``lm`` is the model itself, for ``save_pretrained`` and generation.

    The class is built on first use so that importing this module needs no torch.
    """
    global _CHECKPOINTABLE
    if _CHECKPOINTABLE is None:
        import torch.nn as nn

        class Checkpointable(nn.Module):
            def __init__(self, lm: Any) -> None:
                super().__init__()
                self.lm = lm
                seen: dict[int, str] = {}
                self.tied: dict[str, str] = {}
                for name, tensor in lm.state_dict().items():
                    key = tensor.data_ptr()
                    if key in seen and tensor.numel():
                        self.tied[name] = seen[key]
                    else:
                        seen[key] = name

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return self.lm(*args, **kwargs)

            def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
                return {k: v for k, v in self.lm.state_dict(*args, **kwargs).items()
                        if k not in self.tied}

            def load_state_dict(self, state: Mapping[str, Any],  # type: ignore[override]
                                strict: bool = True) -> Any:
                result = self.lm.load_state_dict(dict(state), strict=False)
                missing = [k for k in result.missing_keys if k not in self.tied]
                if strict and (missing or result.unexpected_keys):
                    raise KeyError(f"checkpoint does not fit the model: missing {missing}, "
                                   f"unexpected {list(result.unexpected_keys)}")
                if any(name in result.missing_keys for name in self.tied):
                    self.lm.tie_weights()
                return result

        _CHECKPOINTABLE = Checkpointable
    return _CHECKPOINTABLE(lm)


# -- the recipe ------------------------------------------------------------------------------------

def build_tool_caller(spec: dict[str, Any], config: dict[str, Any], data: Path,
                      framework: str) -> Built:
    train, holdout, manifest = read_conversations(data)
    if not train:
        raise ValueError(
            f"no conversations under {data}. Expected .jsonl rows with 'messages' (and "
            "'tools'), which ml-stack-train-tools writes.")

    size = config.get("size") or sorted(spec["sizes"])[0]
    base = str(manifest.get("base") or spec["sizes"][size]["base"])
    context = int(config["context"])
    seed = int(config.get("seed") or 0)
    device = device_for()
    model, tokenizer = load_base(base, device=device)

    def rendered(rows: Sequence[Mapping[str, Any]]) -> tuple[list, int]:
        kept, dropped = [], 0
        for row in rows:
            got = render(tokenizer, row["messages"], row.get("tools"), context=context)
            if got is None:
                dropped += 1
            else:
                kept.append(got)
        return kept, dropped

    train_rows, train_dropped = rendered(train)
    holdout_rows, holdout_dropped = rendered(holdout)
    if not train_rows:
        raise ValueError(f"every conversation was cut off at context {context}; raise it")
    answer_tokens = sum(sum(1 for lab in labels if lab != IGNORE) for _, labels in train_rows)
    all_tokens = sum(len(ids) for ids, _ in train_rows)

    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=0.0)

    def loss(m: Any, batch: Mapping[str, np.ndarray]) -> Any:
        tensors = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
        return m(**tensors).loss

    batch_size = int(config["batch_size"])
    pad = int(tokenizer.pad_token_id)
    return Built(
        model=model, optimizer=optimizer, loss=loss,
        batches=conversation_batches(train_rows, batch_size=batch_size, pad_id=pad, seed=seed),
        eval_batches=conversation_batches(holdout_rows or train_rows, batch_size=batch_size,
                                          pad_id=pad, seed=seed + 1),
        config={"recipe": "tool-calls", "size": size, "framework": "torch", "base": base,
                "device": str(device), "rows": len(train) + len(holdout),
                "train_rows": len(train_rows), "holdout_rows": len(holdout_rows),
                "dropped_at_context": train_dropped + holdout_dropped,
                "answer_tokens": answer_tokens, "read_tokens": all_tokens - answer_tokens,
                **config},
    )


def save_pretrained(run_dir: Path | str, base: str, out_dir: Path | str) -> Path:
    """The run's latest checkpoint as a Hugging Face directory, which is what a GGUF converter reads."""
    from ml_stack.train.checkpoint import find_latest, load_tensors

    run_dir, out_dir = Path(run_dir).expanduser(), Path(out_dir).expanduser()
    latest = find_latest(run_dir)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint under {run_dir}; train first")
    from safetensors.torch import load_file

    model, tokenizer = load_base(base, device="cpu")
    model.load_state_dict(load_tensors(latest, read_tensors=lambda p: dict(load_file(str(p)))))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
