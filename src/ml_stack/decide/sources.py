"""Where a pointer decider's files are: the pinned released checkpoint, or a directory that
`ml_stack.train.decider` wrote."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.decide.fetch import Pin, locate
from ml_stack.decide.pins import STRANDS_V19, Checkpoint
from ml_stack.decide.types import DecideError
from ml_stack.deciders import CONFIG
from ml_stack.files import sha256_file
from ml_stack.home import expand

FORMAT = "ml-stack-decider/1"
REFUSED_SUFFIXES = frozenset({".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".npy", ".npz"})


@dataclass(frozen=True, slots=True)
class Source:
    """Everything a pointer decider loads, as local paths."""

    name: str
    base_dir: Path
    tokenizer_dir: Path
    head: Path
    lora_dir: Path | None
    pointer_dim: int
    temperature: float
    dtype: str = "bfloat16"
    details: Any = None


def strands_source(checkpoint: Checkpoint = STRANDS_V19, *, download: bool = False) -> Source:
    """The released checkpoint's files, each verified against its pin."""
    paths = {p.filename: locate(p, download=download) for p in checkpoint.files}
    cfg = json.loads(paths["hobson_config.json"].read_text())
    if cfg.get("head_type") != "pointer":
        raise DecideError(f"{checkpoint.name}: head_type {cfg.get('head_type')!r} is not a "
                          "pointer head")
    return Source(checkpoint.name, paths["config.json"].parent, paths["tokenizer.json"].parent,
                  paths["head.safetensors"], paths["lora/adapter_config.json"].parent,
                  int(cfg["pointer_dim"]),
                  float(cfg.get("temperature_by_kind", {}).get("choice",
                                                               cfg.get("temperature", 1.0))))


def pins_of(rows: list[dict[str, Any]], repo: str, revision: str) -> list[Pin]:
    """`Pin`s from the ``base`` entry of a decider config."""
    return [Pin(repo, revision, r["filename"], r["sha256"], int(r["size"])) for r in rows]


def local_source(path: Path | str, *, download: bool = False) -> Source:
    """A directory written by `ml_stack.train.decider`: refuses pickled files, checks the
    adapter and head against the hashes in its config, and resolves its base by pin, or by
    the local directory it was trained on."""
    root = expand(path)
    found = [p.name for p in root.rglob("*") if p.suffix.lower() in REFUSED_SUFFIXES]
    if found:
        raise DecideError(f"{root} holds files that are not loaded: {sorted(found)}")
    cfg = json.loads((root / CONFIG).read_text())
    if cfg.get("format") != FORMAT:
        raise DecideError(f"{root / CONFIG} is not a {FORMAT} config")
    for name, want in cfg["sha256"].items():
        if not (root / name).is_file() or sha256_file(root / name) != want:
            raise DecideError(f"{root / name} does not match the hash recorded when it was "
                              "written")
    base = cfg["base"]
    if "path" in base:
        base_dir = tok_dir = Path(base["path"])
    else:
        paths = {p.filename: locate(p, download=download)
                 for p in pins_of(base["files"], base["repo"], base["revision"])}
        base_dir, tok_dir = paths["config.json"].parent, paths["tokenizer.json"].parent
    lora = root / "lora"
    return Source(cfg["name"], base_dir, tok_dir, root / "head.safetensors", lora if lora.is_dir() else None,
                  int(cfg["pointer_dim"]), valid_temperature(cfg["temperature"]),
                  cfg.get("dtype", "bfloat16"),
                  cfg)


MAX_TEMPERATURE = 100.0


def valid_temperature(value: Any) -> float:
    """``value`` as a temperature, or `DecideError` unless it is finite and in (0, 100]."""
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise DecideError(f"temperature {value!r} is not a number") from None
    if not 0.0 < t <= MAX_TEMPERATURE:
        raise DecideError(f"temperature {t} is outside (0, {MAX_TEMPERATURE:g}]")
    return t


def question_key(question: str) -> str:
    """A question as the by-question temperature table keys it."""
    return " ".join(question.lower().split())[:300]


def temperature_for(source: Source, question: str) -> float:
    """The temperature for ``question``: the one fitted for its kind in a trained decider's
    config, else the decider's single temperature."""
    cfg = source.details
    if isinstance(cfg, dict):
        kind = cfg.get("question_kinds", {}).get(question_key(question))
        by_kind = cfg.get("temperature_by_kind", {})
        if kind in by_kind:
            return valid_temperature(by_kind[kind])
    return source.temperature
