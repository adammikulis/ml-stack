"""Rows on their way to disk: which half a question falls in, and the files a recipe reads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HOLDOUT_EVERY = 10
"""One seed question in ten is held out, by hash, with every paraphrase of it."""


def _hash(text: str) -> int:
    return int(hashlib.sha256(text.strip().lower().encode()).hexdigest(), 16)


def side_of(seed_question: str) -> str:
    """``"holdout"`` or ``"train"`` for one seed question, the same answer every run.

    A paraphrase goes wherever its original went: a paraphrase in training and its original
    in the holdout would score the memorising, not the calling.
    """
    return "holdout" if _hash(seed_question) % HOLDOUT_EVERY == 0 else "train"


def lines(rows: Sequence[Mapping[str, Any]]) -> str:
    """The rows as JSONL, one object a line."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def split(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(train, holdout)`` as the rows' ``split`` field says."""
    train = [dict(r) for r in rows if r.get("split") != "holdout"]
    holdout = [dict(r) for r in rows if r.get("split") == "holdout"]
    return train, holdout


def counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How many rows call each tool, by name."""
    out: dict[str, int] = {}
    for r in rows:
        out[str(r.get("tool"))] = out.get(str(r.get("tool")), 0) + 1
    return dict(sorted(out.items()))


def write_dataset(out: Path, rows: Sequence[Mapping[str, Any]], *, base: str,
                  **manifest: Any) -> dict[str, Any]:
    """``train.jsonl``, ``holdout.jsonl`` and a ``manifest.json`` naming the base model.

    The manifest is how the ``tool-calls`` recipe knows which chat template to render
    with: a conversation set is made *for* a model, and the recipe reads that here rather
    than asking again.
    """
    out.mkdir(parents=True, exist_ok=True)
    train, holdout = split(rows)
    for name, part in (("train.jsonl", train), ("holdout.jsonl", holdout)):
        (out / name).write_text(lines(part))
    summary = {"base": base, "rows": len(rows), "train": len(train), "holdout": len(holdout),
               "per_tool": counts(rows), **manifest}
    (out / "manifest.json").write_text(json.dumps(summary, indent=2))
    return summary
