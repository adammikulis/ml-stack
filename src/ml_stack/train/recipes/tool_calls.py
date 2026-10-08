"""Fine-tune a chat model to call a project's tools, on conversations made for it.

The data is what ``ml-stack-train-tools`` synthesises — or anything in the same shape: rows
of ``{"messages": [...], "tools": [...]}`` — and the base is a Hugging Face causal LM with
a chat template, ``unsloth/gemma-4-E4B-it`` unless the data's manifest names another.
Each conversation is rendered through the model's own template, so what it learns is the
exact format it will be served with, and the loss is on the assistant turn only: the
question and the tool declarations are read, never predicted.

This is torch whatever the machine's default backend is, because the checkpoint decides:
a Hugging Face safetensors model loads through ``transformers``, which is torch. The
device is the machine's accelerator unless ``ML_STACK_DEVICE`` says otherwise (``cpu`` in
the tests, so a tiny model never touches a GPU that a benchmark may be using).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ml_stack.train.recipes.base import resolve_base
from ml_stack.train.recipes.built import Built
from ml_stack.train.recipes.conversations import (
    IGNORE,
    conversation_batches,
    read_conversations,
    render,
)
from ml_stack.train.recipes.tool_calls_mlx import build as build_mlx


def device_for() -> Any:
    """The torch device a fine-tune runs on: ``ML_STACK_DEVICE`` if set, else the accelerator.

    A string naming the gap when torch is absent, so a ``--dry-run`` can still print its
    plan on a machine that has yet to install it.
    """
    from ml_stack.backend.device import resolve_torch_device
    try:
        return resolve_torch_device(os.environ.get("ML_STACK_DEVICE") or None)
    except (ImportError, RuntimeError):
        return "torch (not installed)"


# -- the model ------------------------------------------------------------------------------------

def load_base(base: str, *, device: Any = None, dtype: Any = None) -> tuple[Any, Any]:
    """``(model, tokenizer)`` for a Hugging Face causal LM, on ``device``.

    Whatever ``transformers`` cannot import is reported as the seam it is, rather than as
    a missing attribute three frames down. ``dtype`` is float32 unless a caller says
    otherwise: a full fine-tune updates these weights and wants the width, while a LoRA
    freezes them and asks for bf16, which is the difference between 32G and 16G of base
    for an 8B model.
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
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype or torch.float32)
    model.config.use_cache = False
    model.to(device or device_for())
    return model, tokenizer


def frozen_dtype(device: Any) -> Any:
    """The width a *frozen* base is held at: bf16 on an accelerator, float32 on the CPU.

    bf16 matmul on the CPU is emulated and slower than the float32 it saves memory over,
    and a machine running the tests has memory to spare; on MPS or CUDA it halves the
    resident base and is what the adapter is trained against anyway.
    """
    import torch

    return torch.float32 if str(device).startswith("cpu") else torch.bfloat16


# -- the recipe ------------------------------------------------------------------------------------

def build_tool_caller(spec: dict[str, Any], config: dict[str, Any], data: Path | None,
                      framework: str) -> Built:
    if data is None:
        raise ValueError("tool-calls is built from its conversations; it has no build without "
                         "data, and its trained model is read with save_pretrained")
    if framework == "mlx":
        return build_mlx(spec, config, data)
    train, holdout, manifest = read_conversations(data)
    if not train:
        raise ValueError(
            f"no conversations under {data}. Expected .jsonl rows with 'messages' (and "
            "'tools'), which ml-stack-train-tools writes.")

    size = config.get("size") or sorted(spec["sizes"])[0]
    base, _ = resolve_base(spec, config, manifest)
    context = int(config["context"])
    seed = int(config.get("seed") or 0)
    device = device_for()
    wants_lora = bool(config.get("lora"))
    model, tokenizer = load_base(base, device=device,
                                 dtype=frozen_dtype(device) if wants_lora else None)

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

    def loss(m: Any, batch: Mapping[str, np.ndarray]) -> Any:
        tensors = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
        return m(**tensors).loss

    extra: dict[str, Any] = {}
    step = None
    if wants_lora:
        from ml_stack.train.lora import Lora, LoraStep, attach, trainable_parameters

        settings = Lora.of(config)
        model = attach(model, settings)
        trainable = trainable_parameters(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=float(config["learning_rate"]), weight_decay=0.0)
        step = LoraStep(model, optimizer, loss)
        extra = {"lora_targets_used": list(settings.targets),
                 "trainable_parameters": trainable,
                 "base_dtype": str(next(model.parameters()).dtype)}
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                      weight_decay=0.0)

    batch_size = int(config["batch_size"])
    pad = int(tokenizer.pad_token_id)
    return Built(
        model=model, optimizer=optimizer, loss=loss, step=step,
        batches=conversation_batches(train_rows, batch_size=batch_size, pad_id=pad, seed=seed),
        eval_batches=conversation_batches(holdout_rows or train_rows, batch_size=batch_size,
                                          pad_id=pad, seed=seed + 1),
        config={**config, "recipe": "tool-calls", "size": size, "framework": "torch",
                "base": base,
                "device": str(device), "rows": len(train) + len(holdout),
                "train_rows": len(train_rows), "holdout_rows": len(holdout_rows),
                "dropped_at_context": train_dropped + holdout_dropped,
                "answer_tokens": answer_tokens, "read_tokens": all_tokens - answer_tokens,
                **extra},
    )


def save_pretrained(run_dir: Path | str, base: str, out_dir: Path | str) -> Path:
    """The run's latest checkpoint as a Hugging Face directory, which is what a GGUF converter reads."""
    from ml_stack.train.checkpoint import find_latest, load_tensors
    from ml_stack.train.step import load_state_once

    run_dir, out_dir = Path(run_dir).expanduser(), Path(out_dir).expanduser()
    latest = find_latest(run_dir)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint under {run_dir}; train first")
    from ml_stack.train.checkpoint import load_state

    if load_state(latest).config.get("lora"):
        raise ValueError(
            f"{latest} is a LoRA checkpoint: it holds the adapter, not the whole model, so "
            "there is nothing here to save as a Hugging Face directory. Merge it into the "
            "base instead -- ml_stack.train.lora.merge, which `ml-stack-train-run "
            "--export-gguf` runs for you.")
    from safetensors.torch import load_file

    model, tokenizer = load_base(base, device="cpu")
    load_state_once(model, load_tensors(latest, read_tensors=lambda p: dict(load_file(str(p)))))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
