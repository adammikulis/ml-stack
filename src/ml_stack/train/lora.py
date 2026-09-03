"""LoRA over the ``tool-calls`` recipe: what makes an 8B tool caller trainable here.

The recipe's full fine-tune is right for a 270m base and impossible for an 8B one: bf16
weights, gradients and two Adam moments for eight billion parameters is ~128G of state
before a single activation. A LoRA trains ~0.5% of that -- two small matrices on each
attention and MLP projection -- with the base frozen in bf16, which is 16G of weights and
an optimizer small enough to ignore.

Everything peft-shaped is behind ``require_peft``: the extra is ``ml-stack[train-lora]``,
and without it the recipe says so rather than failing on an attribute three frames down.

What this module owns, end to end:

* ``Lora`` -- rank, alpha, dropout and which projections, read off a validated config.
* ``attach`` / ``LoraStep`` -- the peft wrapper, and a training step whose *checkpoints are
  the adapter*. `TorchStep.parameters` is the model's whole state dict, which for E4B is
  16G written every ``checkpoint_every`` steps; the adapter is ~80M.
* ``plan`` -- what fits and what it will cost, before anything is downloaded, refused over
  the same 30-minute ceiling the bench uses unless the caller said ``--yes``.
* ``merge`` / ``export_gguf`` / ``preflight_export`` -- the adapter folded back into the
  base, through ``ml_stack.gguf.export``, ending at a file ``ml-stack-serve up`` can load
  and ``ml_stack.serve.preflight`` has already agreed with.
* ``fingerprint`` -- the data's hash and example count, for the run's manifest: a fine-tune
  whose training data cannot be identified afterwards cannot be reproduced or believed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.train.step import TorchStep

__all__ = ["CEILING_ENV", "CEILING_MIN", "DEFAULT_TARGETS", "Fit", "Lora", "LoraStep",
           "OverCeiling", "PEFT_MISSING", "attach", "adapter_tensors", "converter",
           "export_gguf", "fingerprint", "merge", "parameters_b", "plan",
           "preflight_export", "quantizer", "refuse_over_ceiling", "require_peft",
           "save_adapter", "span", "summarise", "trainable_parameters"]


# The projections a LoRA is put on. Attention alone is the older recipe and measurably
# weaker on instruction data; the MLP projections are where a Gemma-shaped model keeps most
# of its parameters, and a tool caller is being taught what to *say*, not only what to
# attend to. Names as transformers spells them, which is what peft matches on.
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")

PEFT_MISSING = (
    "a LoRA needs peft, which is not installed: pip install 'ml-stack[train-lora]' "
    "(or pip install peft). Without it the tool-calls recipe is a full fine-tune, which "
    "works for the 270m base and needs ~128G of optimizer state for an 8B one."
)

# Over this many minutes a training run is refused unless --yes. The same rule and the same
# number as `ml_stack.graph.bench.estimate.CEILING_MIN`, kept here rather than imported so
# that starting a fine-tune does not drag the whole bench package in: a training run is
# hours where a bench run is minutes, which is exactly why it must be a decision.
CEILING_MIN = 30.0
CEILING_ENV = "MLSTACK_TRAIN_CEILING"

# Tokens a second, per billion *active* parameters, for a LoRA fine-tune through
# transformers. Derived rather than measured: forward and backward through a LoRA'd layer
# costs about 4 FLOPs per active parameter per token (there is no weight gradient for the
# frozen base), and torch's MPS backend sustains on the order of 2 TFLOP/s of bf16 matmul
# on this class of machine -- so ~500 tokens a second for each billion active parameters.
# It is an estimate and says so everywhere it is printed. The way to replace it with a
# measurement is `ml-stack-train-run --dry-run`, which trains 20 real steps and reports
# seconds per step; `Fit.measured` re-states the plan from that number.
TOKENS_PER_S_PER_B = {"mps": 500.0, "cuda": 4000.0, "cpu": 40.0}

# A frozen base is held in bf16; the adapter is fp32 with two fp32 Adam moments beside it.
BASE_BYTES_PER_PARAM = 2
ADAPTER_BYTES_PER_PARAM = 12
# Activations kept for the backward pass, per token, per billion active parameters. Same
# order of derivation as the throughput above: a Gemma-shaped 4B-active model at batch 4
# and context 2048 keeps a little over 2G of them.
ACT_BYTES_PER_TOKEN_PER_B = 70_000


class OverCeiling(RuntimeError):
    """The estimated wall clock is over the ceiling and nobody said ``--yes``."""


# -- the settings ------------------------------------------------------------------------


@dataclass(frozen=True)
class Lora:
    """Which weights get an adapter, and how big it is."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    targets: tuple[str, ...] = DEFAULT_TARGETS

    @classmethod
    def of(cls, config: Mapping[str, Any]) -> Lora:
        """The LoRA settings out of a validated recipe config."""
        targets = config.get("lora_targets") or ""
        names = tuple(t.strip() for t in str(targets).replace(",", " ").split() if t.strip())
        return cls(rank=int(config.get("lora_rank") or 16),
                   alpha=int(config.get("lora_alpha") or 32),
                   dropout=float(config.get("lora_dropout") or 0.0),
                   targets=names or DEFAULT_TARGETS)

    def said(self) -> str:
        return (f"rank {self.rank}, alpha {self.alpha}, dropout {self.dropout:g}, on "
                + ", ".join(self.targets))

    def as_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "alpha": self.alpha, "dropout": self.dropout,
                "targets": list(self.targets)}


def require_peft() -> Any:
    """The ``peft`` module, or a ``ValueError`` that names the extra to install."""
    try:
        import peft
    except ImportError as exc:                                  # pragma: no cover - env
        raise ValueError(PEFT_MISSING) from exc
    return peft


def attach(model: Any, lora: Lora) -> Any:
    """``model`` with LoRA adapters on ``lora.targets``, everything else frozen."""
    peft = require_peft()
    config = peft.LoraConfig(r=lora.rank, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                             target_modules=list(lora.targets), bias="none",
                             task_type="CAUSAL_LM")
    wrapped = peft.get_peft_model(model, config)
    if not trainable_parameters(wrapped):
        raise ValueError(
            f"no module of this model is named any of {', '.join(lora.targets)}, so the "
            "adapter has nothing to train. Pass --lora-targets with this architecture's "
            "own projection names.")
    return wrapped


def trainable_parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adapter_tensors(model: Any) -> dict[str, Any]:
    """Just the adapter's tensors -- what a LoRA checkpoint is."""
    from peft import get_peft_model_state_dict

    return dict(get_peft_model_state_dict(model))


class LoraStep(TorchStep):
    """A torch step whose checkpoints hold the adapter and not the frozen base.

    The base is 16G for E4B and identical at every step; writing it every
    ``checkpoint_every`` steps would spend the disk and the wall clock on a copy of
    something already on this machine. Restoring is strict on the adapter's own names: a
    checkpoint from a different rank or different target modules does not fit, and a
    partial restore of an adapter is a silently different model.
    """

    name = "torch"

    def parameters(self) -> dict[str, Any]:
        return {k: v.detach().cpu() for k, v in adapter_tensors(self.model).items()}

    def restore(self, tensors: dict[str, Any], optimizer: dict[str, Any] | None) -> None:
        from peft import set_peft_model_state_dict

        from ml_stack.train.checkpoint import CheckpointError

        here = adapter_tensors(self.model)
        missing = sorted(set(here) - set(tensors))
        unexpected = sorted(set(tensors) - set(here))
        resized = [name for name in here.keys() & tensors.keys()
                   if tuple(here[name].shape) != tuple(tensors[name].shape)]
        if missing or unexpected or resized:
            detail = (f"missing {missing[:3]}, unexpected {unexpected[:3]}" if
                      (missing or unexpected) else
                      f"{resized[0]} is {tuple(tensors[resized[0]].shape)} here and "
                      f"{tuple(here[resized[0]].shape)} in the model")
            raise CheckpointError(
                f"this adapter checkpoint does not fit the model: {detail}. A LoRA "
                "checkpoint only fits the rank and the target modules it was trained "
                "with; train a different rank into a different --out.")
        set_peft_model_state_dict(self.model, tensors)
        if not optimizer:
            return
        by_name = dict(self.model.named_parameters())
        for flat, value in optimizer.items():
            base, _, key = flat.rpartition(".")
            param = by_name.get(base)
            if param is None:
                continue
            self.opt.state.setdefault(param, {})[key] = value


def save_adapter(model: Any, out: Path | str) -> Path:
    """The adapter alone, in peft's own layout, which ``merge`` reads back."""
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    if not (out / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(
            f"peft wrote no adapter_model.safetensors under {out}; there is nothing to merge")
    return out


def merge(base: str, adapter: Path | str, out: Path | str, *, device: str = "cpu") -> Path:
    """``base`` with ``adapter`` folded into its weights, in Hugging Face layout.

    On the CPU by default: merging is one pass of arithmetic over the weights and wants
    memory rather than a GPU, and the GPU is usually the thing a bench is holding. In
    float32, because the merged weights are what the converter reads and a bf16 merge
    rounds every one of them -- 32G resident and 32G written for an 8B base, both
    transient, and the export deletes its intermediate afterwards.
    """
    require_peft()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter, out = Path(adapter).expanduser(), Path(out).expanduser()
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"no adapter under {adapter}; train one first")

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model.to(device)
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(base).save_pretrained(str(out))
    return out


# -- what it will cost -------------------------------------------------------------------


def managed_source() -> Path:
    """The managed llama.cpp build's source checkout -- where its converter lives.

    Read off ``serve.build.SRC_DIR`` when that import works, so there is one answer to
    "which llama.cpp is this machine's", and the same path spelled out when it does not.
    """
    try:
        from ml_stack.serve.build import SRC_DIR

        return Path(SRC_DIR)
    except Exception:                                       # noqa: BLE001 - a bare install
        return Path.home() / ".ml-stack" / "llama.cpp" / "src"


def ceiling_default() -> float:
    """The ceiling in minutes: the environment's, else `CEILING_MIN`."""
    try:
        return float(os.environ.get(CEILING_ENV, "") or CEILING_MIN)
    except ValueError:
        return CEILING_MIN


def span(seconds: float) -> str:
    """``45 s``, ``26 min``, ``2 h 10 min`` -- the shapes the bench's estimates print."""
    whole = int(round(seconds))
    if whole < 60:
        return f"{whole} s"
    minutes = int(round(whole / 60))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def parameters_b(base: str, size_spec: Mapping[str, Any] | None = None) -> tuple[float, float]:
    """``(parameters, active parameters)`` in billions, without loading anything.

    The recipe's size entry when there is one -- it is written down precisely so a plan can
    be printed before 16G is downloaded -- else the safetensors on disk for a local
    directory, at the width its config names. ``(0, 0)`` when neither says, which is a plan
    with no wall clock in it rather than a guessed one.
    """
    spec = dict(size_spec or {})
    params = float(spec.get("params_m") or 0) / 1000
    active = float(spec.get("active_m") or 0) / 1000 or params
    if params:
        return params, active

    path = Path(base).expanduser()
    if not path.is_dir():
        return 0.0, 0.0
    weights = sum(f.stat().st_size for f in path.glob("*.safetensors"))
    if not weights:
        return 0.0, 0.0
    width = 4
    try:
        config = json.loads((path / "config.json").read_text())
        width = {"float32": 4, "bfloat16": 2, "float16": 2}.get(
            str(config.get("dtype") or config.get("torch_dtype") or "float32"), 4)
    except (OSError, json.JSONDecodeError):
        pass
    counted = weights / width / 1e9
    return counted, counted


@dataclass
class Fit:
    """What a LoRA run needs and what it should take, said before it is started."""

    base: str
    device: str
    params_b: float
    active_b: float
    lora: Lora
    batch: int
    context: int
    steps: int
    examples: int
    trainable_b: float = 0.0
    ceiling_min: float = CEILING_MIN
    seconds_per_step: float = 0.0
    measured: bool = False

    @property
    def tokens_per_step(self) -> int:
        """At most: rows are padded to the longest in their batch, never past ``context``."""
        return self.batch * self.context

    @property
    def seconds(self) -> float:
        return self.seconds_per_step * self.steps

    @property
    def epochs(self) -> float:
        return (self.steps * self.batch / self.examples) if self.examples else 0.0

    @property
    def resident_gb(self) -> float:
        base = self.params_b * 1e9 * BASE_BYTES_PER_PARAM
        adapter = self.trainable_b * 1e9 * ADAPTER_BYTES_PER_PARAM
        activations = self.tokens_per_step * self.active_b * ACT_BYTES_PER_TOKEN_PER_B
        return (base + adapter + activations) / 1e9

    @property
    def over(self) -> bool:
        return self.seconds > self.ceiling_min * 60

    def lines(self) -> list[str]:
        how = "measured" if self.measured else "estimated"
        out = [f"lora: {self.lora.said()}"
               + (f"; {self.trainable_b * 1000:.0f}M trainable" if self.trainable_b else ""),
               f"plan: {self.base} on {self.device}, batch {self.batch} × context "
               f"{self.context} = {self.tokens_per_step} tokens a step, {self.steps} steps"
               + (f" over {self.examples} examples ({self.epochs:.2f} epochs)"
                  if self.examples else "")]
        if self.params_b:
            out.append(f"fit: {self.params_b:.1f}B parameters "
                       f"({self.active_b:.1f}B active) ≈ {self.resident_gb:.1f}G resident "
                       "-- frozen base in bf16, adapter and its Adam moments, activations")
        if self.seconds_per_step:
            out.append(f"cost: {how} {self.seconds_per_step:.1f} s/step, "
                       f"{span(self.seconds)} in all"
                       + ("" if self.measured else
                          " (an estimate; --dry-run trains 20 real steps and measures it)")
                       + (" -- over the ceiling" if self.over else ""))
        else:
            out.append("cost: no parameter count for this base, so no estimate; "
                       "the ceiling cannot refuse what it cannot estimate")
        return out

    def refusal(self) -> str:
        return (f"error: estimated {span(self.seconds)}, over the {self.ceiling_min:g} min "
                f"ceiling -- a fine-tune is hours, so it is a decision and not an accident. "
                f"Train fewer steps (--set steps=N), raise the ceiling (--ceiling MINUTES, "
                f"or {CEILING_ENV}), measure first (--dry-run), or pass --yes.")

    def as_dict(self) -> dict[str, Any]:
        return {"base": self.base, "device": self.device, "batch": self.batch,
                "context": self.context, "steps": self.steps, "examples": self.examples,
                "params_b": round(self.params_b, 3), "active_b": round(self.active_b, 3),
                "trainable_m": round(self.trainable_b * 1000, 1),
                "resident_gb": round(self.resident_gb, 1),
                "seconds_per_step": round(self.seconds_per_step, 3),
                "estimated_seconds": round(self.seconds, 1), "measured": self.measured,
                "lora": self.lora.as_dict()}


def plan(config: Mapping[str, Any], *, base: str, device: str, examples: int = 0,
         size_spec: Mapping[str, Any] | None = None, trainable: int = 0,
         ceiling_min: float | None = None, seconds_per_step: float = 0.0) -> Fit:
    """What this run needs and what it should take. Nothing here loads a model.

    ``seconds_per_step`` when a ``--dry-run`` has already measured it; otherwise it is
    derived from the active parameter count and `TOKENS_PER_S_PER_B`.
    """
    lora = Lora.of(config)
    params, active = parameters_b(base, size_spec)
    batch, context = int(config["batch_size"]), int(config["context"])
    steps = int(config["steps"])
    kind = str(device).split(":")[0] or "cpu"
    rate = TOKENS_PER_S_PER_B.get(kind, TOKENS_PER_S_PER_B["cpu"])

    per_step = seconds_per_step
    if not per_step and active:
        per_step = batch * context * active / rate

    fit = Fit(base=base, device=str(device), params_b=params, active_b=active, lora=lora,
              batch=batch, context=context, steps=steps, examples=examples,
              trainable_b=trainable / 1e9,
              ceiling_min=ceiling_default() if ceiling_min is None else float(ceiling_min),
              seconds_per_step=per_step, measured=bool(seconds_per_step))
    if not trainable and params:
        # rank r on each of t projections costs r × (in + out) per module; for a Gemma-shaped
        # model that lands near half a percent of the weights at rank 16.
        fit.trainable_b = params * 0.005 * (lora.rank / 16)
    return fit


# -- the data behind it ------------------------------------------------------------------


def fingerprint(data: Path | str) -> dict[str, Any]:
    """The training data's identity: every ``.jsonl`` under it, hashed, and how many rows.

    One digest over the files in name order, plus each file's own, so two runs can be said
    to have trained on the same data or not. A fine-tune is a claim about data; a claim
    whose data cannot be identified afterwards is not one anybody can check.
    """
    path = Path(data).expanduser()
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    whole = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    examples = 0
    for f in files:
        if not f.is_file():
            continue
        digest = hashlib.sha256()
        count = 0
        with f.open("rb") as handle:
            for block in iter(lambda h=handle: h.read(1 << 20), b""):
                digest.update(block)
                whole.update(block)
        with f.open("r", errors="replace") as handle:
            count = sum(1 for line in handle if line.strip())
        examples += count
        rows.append({"file": f.name, "sha256": digest.hexdigest(), "rows": count})
    return {"path": str(path), "sha256": whole.hexdigest(), "examples": examples,
            "files": rows}


# -- out the far end ---------------------------------------------------------------------


def converter(explicit: Path | str | None = None) -> Path:
    """llama.cpp's ``convert_hf_to_gguf.py``, the managed build's copy preferred.

    ``ml-stack-serve build`` keeps a source checkout at ``~/.ml-stack/llama.cpp/src`` and
    builds the server this machine actually serves with out of it. Converting through that
    same checkout is how the exported GGUF is written by the code that will read it -- a
    converter from an older clone writes metadata a newer server has stopped reading, and
    the failure arrives at the far end of a load. An explicit path and ``$LLAMA_CPP_ROOT``
    still win, because they are somebody saying which one they mean.
    """
    from ml_stack.gguf.tools import CONVERTER_NAME, require_converter

    if explicit:
        return Path(explicit).expanduser().resolve()
    for key in ("LLAMA_CPP_ROOT", "LLAMA_CPP_DIR"):
        if root := os.environ.get(key):
            candidate = Path(root).expanduser() / CONVERTER_NAME
            if candidate.is_file():
                return candidate.resolve()
    managed = managed_source() / CONVERTER_NAME
    if managed.is_file():
        return managed.resolve()
    return require_converter()


def quantizer(explicit: Path | str | None = None) -> Path:
    """``llama-quantize``, the managed build's own before anything on PATH."""
    from ml_stack.gguf.tools import require_quantize

    if explicit:
        return Path(explicit).expanduser().resolve()
    root = managed_source().parent
    for candidate in (root / "current" / "llama-quantize",
                      managed_source() / "build" / "bin" / "llama-quantize"):
        if candidate.is_file():
            return candidate.resolve()
    return require_quantize()


def export_gguf(model_dir: Path | str, out_dir: Path | str, *, name: str,
                quant: str = "Q8_0", convert_with: Path | str | None = None,
                quantize_with: Path | str | None = None) -> Any:
    """The merged model as a GGUF, through the existing export path.

    ``ml_stack.gguf.export`` is the whole of it -- convert, leave the tokenizer metadata
    alone, quantise -- handed the managed build's tools rather than left to search for a
    clone of its own.
    """
    from ml_stack.gguf import export

    return export(model_dir, out_dir, name=name, quant=quant, fix_space_prefix=None,
                  converter=converter(convert_with), quantizer=quantizer(quantize_with))


def preflight_export(gguf: Path | str, *, binary: Path | str | None = None,
                     context: int = 4096, **seams: Any) -> Any:
    """Ask the serve path's own questions of the file just written, before serving it.

    A `Report`, from `ml_stack.serve.preflight.Preflight`: are the shards there, is the
    architecture one this build reads, does it fit, are the flags a spec would emit ones
    this build accepts. No process is started and no tensor is read -- a GGUF header is a
    few hundred bytes off the front of the file. ``binary`` defaults to whatever
    ``ml-stack-serve`` would use; the ``seams`` are `Preflight`'s own, for a test that
    hands in the facts.
    """
    from ml_stack.serve.backend import ServerSpec
    from ml_stack.serve.preflight import Preflight

    if binary is None:
        from ml_stack.serve.binary import find_binary

        binary = find_binary()
    spec = ServerSpec(model=str(Path(gguf).expanduser()), context=context)
    return Preflight(spec, binary=binary, **seams)


def summarise(report: Any) -> str:
    """A report's failing lines, or the count when every one passed."""
    if report.ok:
        return f"{len(report.checks)} checks, all ok"
    return "; ".join(f"{c.name}: {c.detail}" for c in report.checks if not c.ok)


def refuse_over_ceiling(fit: Fit, *, yes: bool) -> None:
    """Raise `OverCeiling` unless the run is short enough or somebody said ``--yes``."""
    if fit.over and not yes:
        raise OverCeiling(fit.refusal())


