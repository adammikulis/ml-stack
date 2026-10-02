"""``ml-stack-train-decider``: fine-tune a pointer-head decision model from labelled cases.

    ml-stack-train-decider guards --out ./my-decider --steps 200
    ml-stack-train-decider cases.jsonl --out ./my-decider --init strands --download

A LoRA on a small base model plus a ~1M-parameter pointer head, trained on the cases with
the options shuffled on every pass, then a temperature fitted on cases held apart from both
training and scoring. The output directory holds ``head.safetensors``, ``lora/``, the
tokenizer files, ``decider.json`` (base pins, file hashes, temperature), ``manifest.json``
(data hash, split sizes, metrics) and ``model_card.md``, and `PointerDecider` loads it.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.command import Group, flag, option
from ml_stack.decide import pointer_prompt, registry
from ml_stack.decide.base import text_of
from ml_stack.decide.calibrate import fit_temperature, softmax
from ml_stack.decide.cases import Case, fingerprint, read_cases
from ml_stack.decide.eval import brier, ece, top_of
from ml_stack.decide.fetch import locate
from ml_stack.decide.guards import guard_cases
from ml_stack.decide.pins import QWEN35_0_8B_BASE, STRANDS_V19, Checkpoint
from ml_stack.decide.pointer import build_head, device_name, load_torso
from ml_stack.decide.sources import CONFIG, FORMAT, Source, strands_source
from ml_stack.decide.types import DecideError
from ml_stack.files import sha256_file, write_json
from ml_stack.home import expand
from ml_stack.log import say, warn
from ml_stack.train.holdout import by_group
from ml_stack.train.lora import (
    DEFAULT_TARGETS,
    Lora,
    require_peft,
    trainable_parameters,
)
from ml_stack.train.schedule import warmup_cosine
from ml_stack.train.step import TorchStep
from ml_stack.train.trainer import Trainer

__all__ = ["COMMANDS", "Result", "Settings", "main", "train"]

SMALL_DATA = 500
"""Training cases under which the model card says the run checks the pipeline and nothing else."""

GROUP_KEY = "id"


@dataclass(frozen=True)
class Settings:
    """What a run trains, on what, and for how long."""

    name: str = "decider"
    base: Checkpoint | Path = QWEN35_0_8B_BASE
    init: Checkpoint | None = None
    steps: int = 60
    batch_size: int = 4
    lr: float = 2e-4
    head_lr_scale: float = 5.0
    lora: Lora = field(default_factory=lambda: Lora(16, 32, 0.05, DEFAULT_TARGETS))
    seed: int = 0
    device: str = "auto"
    dtype: str = "bfloat16"
    max_tokens: int = 1024
    calibrate_fraction: float = 0.15
    test_fraction: float = 0.25
    download: bool = False


@dataclass
class Result:
    """What a run produced: where, the splits, the metrics and the loss history."""

    out: Path
    n_train: int
    n_calibrate: int
    n_test: int
    temperature: float
    metrics: dict[str, float]
    baseline: dict[str, float] | None
    losses: list[float]
    seconds: float
    data_hash: str


def _torch() -> Any:
    import torch
    return torch


def encode(tok: Any, case: Case, order: Sequence[int], max_tokens: int) -> dict[str, Any]:
    """One case rendered with its options in ``order``: token ids and the scoring positions."""
    options = tuple(case.options[i] for i in order)
    text = pointer_prompt.render(case.question, _state_text(case), options)
    enc = tok(text.text, return_offsets_mapping=True, add_special_tokens=False)
    if len(enc["input_ids"]) > max_tokens:
        raise DecideError(f"case {case.id or case.question[:30]!r} is {len(enc['input_ids'])} "
                          f"tokens; the limit is {max_tokens}")
    spots = pointer_prompt.positions([tuple(o) for o in enc["offset_mapping"]], text.spans)
    return {"ids": enc["input_ids"], "spots": spots, "order": list(order)}


def _state_text(case: Case) -> str:
    return text_of(case.state)


def collate(torch: Any, rows: list[dict[str, Any]], labels: list[int], device: str) -> dict[str, Any]:
    """Right-padded tensors for ``rows``."""
    width = max(len(r["ids"]) for r in rows)
    k = max(len(r["spots"]) for r in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    spots = torch.zeros(len(rows), k, dtype=torch.long)
    valid = torch.zeros(len(rows), k, dtype=torch.bool)
    for i, r in enumerate(rows):
        n = len(r["ids"])
        ids[i, :n] = torch.tensor(r["ids"])
        mask[i, :n] = 1
        spots[i, :len(r["spots"])] = torch.tensor(r["spots"])
        valid[i, :len(r["spots"])] = True
    last = mask.sum(1) - 1
    return {"ids": ids.to(device), "mask": mask.to(device), "spots": spots.to(device),
            "valid": valid.to(device), "last": last.to(device),
            "labels": torch.tensor(labels).to(device)}


def attach_features(model: Any, lora: Lora) -> Any:
    """``model`` with LoRA adapters on ``lora.targets``, for a decoder with no language-model
    head (peft's ``FEATURE_EXTRACTION``)."""
    peft = require_peft()
    config = peft.LoraConfig(r=lora.rank, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                             target_modules=list(lora.targets), bias="none",
                             task_type="FEATURE_EXTRACTION")
    return peft.get_peft_model(model, config)


def build_net(torch: Any, torso: Any, head: Any) -> Any:
    """The module a run trains: torso, then the pointer head over its hidden states."""
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.torso, self.head = torso, head

        def forward(self, batch: dict[str, Any]) -> Any:
            """Logits [B, K] with positions past a case's option count at -inf."""
            hidden = self.torso(input_ids=batch["ids"], attention_mask=batch["mask"],
                                use_cache=False).last_hidden_state
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            query = hidden[rows, batch["last"]].float()
            gathered = hidden[rows.unsqueeze(1), batch["spots"]].float()
            logits = self.head(query, gathered)
            return logits.masked_fill(~batch["valid"], float("-inf"))

    return Net()


class DeciderStep(TorchStep):
    """A torch step that sets each group's rate from its ``scale`` and checkpoints only what
    trains."""

    name = "torch"

    def learning_rate(self, lr: float) -> None:
        for group in self.opt.param_groups:
            group["lr"] = lr * group.get("scale", 1.0)

    def parameters(self) -> dict[str, Any]:
        return {k: v.detach().cpu() for k, v in self.model.named_parameters() if v.requires_grad}


def _loss(model: Any, batch: dict[str, Any]) -> Any:
    torch = _torch()
    return torch.nn.functional.cross_entropy(model(batch), batch["labels"])


def predict(net: Any, tok: Any, cases: Sequence[Case], settings: Settings, device: str
            ) -> list[list[float]]:
    """Raw logits for every case in its own option order, in batches of ``batch_size``."""
    torch = _torch()
    net.eval()
    out: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(cases), settings.batch_size):
            part = cases[start:start + settings.batch_size]
            rows = [encode(tok, c, range(len(c.options)), settings.max_tokens) for c in part]
            logits = net(collate(torch, rows, [c.label_index for c in part], device))
            out.extend([x for x in row if x != float("-inf")] for row in logits.tolist())
    return out


def metrics_of(logits: Sequence[Sequence[float]], cases: Sequence[Case], temperature: float
               ) -> dict[str, float]:
    """Accuracy, Brier, ECE and mean confidence of ``logits`` scaled by ``temperature``."""
    rows = [softmax([x / temperature for x in r]) for r in logits]
    labels = [c.label_index for c in cases]
    return {"n": len(cases),
            "accuracy": sum(top_of(r) == y for r, y in zip(rows, labels, strict=True)) / len(rows),
            "brier": brier(rows, labels), "ece": ece(rows, labels),
            "confidence": sum(max(r) for r in rows) / len(rows)}


def splits(cases: Sequence[Case], settings: Settings) -> tuple[list[Case], list[Case], list[Case]]:
    """Train, calibration and test cases, split by whole groups so no group is in two."""
    groups = [c.group or c.id or c.question for c in cases]
    rest = by_group(cases, groups, settings.test_fraction, seed=settings.seed)
    inner = by_group(rest.train, [c.group or c.id or c.question for c in rest.train],
                     settings.calibrate_fraction / (1 - settings.test_fraction),
                     seed=settings.seed + 1)
    return list(inner.train), list(inner.holdout), list(rest.holdout)


def _write_card(out: Path, cfg: dict[str, Any], result: Result) -> None:
    small = result.n_train < SMALL_DATA
    m, b = result.metrics, result.baseline
    lines = [f"# {cfg['name']}", "",
             "A pointer-head decision model: a LoRA on "
             f"`{cfg['base']['repo']}` and a pointer head, trained by `ml-stack-train-decider`.",
             ""]
    if small:
        lines += [f"**Pipeline check, not a quality claim.** Trained on {result.n_train} cases "
                  f"for {cfg['training']['steps']} steps.", ""]
    lines += ["## Data", f"- data hash (SHA-256 of the cases): `{result.data_hash}`",
              f"- cases: {result.n_train} train, {result.n_calibrate} calibration, "
              f"{result.n_test} test (whole groups; no group is in two)",
              "- the licence of the data is the licence of the cases you trained on", "",
              "## Results on the test cases",
              "| | accuracy | Brier | ECE |", "|---|---|---|---|"]
    if b:
        lines.append(f"| before training | {b['accuracy']:.3f} | {b['brier']:.3f} | "
                     f"{b['ece']:.3f} |")
    lines += [f"| after training, temperature {result.temperature:.3f} | {m['accuracy']:.3f} | "
              f"{m['brier']:.3f} | {m['ece']:.3f} |", "",
              "Brier is the sum over options of the squared error, 0 to 2; ECE uses 10 bins.", "",
              "## Files",
              "`head.safetensors`, `lora/adapter_model.safetensors`, `decider.json`, "
              "`manifest.json`. The base model is "
              + (f"the local directory {cfg['base']['path']}." if "path" in cfg["base"] else
                 f"fetched by pinned hash ({cfg['base']['repo']} at "
                 f"{cfg['base']['revision'][:12]})."), "",
              "## Licence",
              f"The base model is {cfg['base']['licence']}. Load with "
              "`PointerDecider(path)`; nothing in this directory is executed or unpickled.", ""]
    (out / "model_card.md").write_text("\n".join(lines), encoding="utf-8")


def _base_entry(given: Checkpoint | Path) -> dict[str, Any]:
    if isinstance(given, Path):
        return {"path": str(expand(given).resolve()), "repo": given.name,
                "licence": "not recorded: a local directory"}
    chosen = given.base_files
    return {"repo": chosen[0].repo, "revision": chosen[0].revision, "licence": given.licence,
            "files": [{"filename": p.filename, "sha256": p.sha256, "size": p.size}
                      for p in chosen]}


def _save(out: Path, net: Any, tok_dir: Path, settings: Settings, result: Result) -> None:
    given = settings.init or settings.base
    from safetensors.torch import save_file
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in net.head.state_dict().items()},
              str(out / "head.safetensors"))
    net.torso.save_pretrained(str(out / "lora"))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(tok_dir / name, out / name)
    hashes = {n: sha256_file(out / n) for n in ("head.safetensors",
                                                 "lora/adapter_model.safetensors",
                                                 "lora/adapter_config.json")}
    cfg = {"format": FORMAT, "name": settings.name, "base": _base_entry(given),
           "pointer_dim": 256, "temperature": result.temperature, "dtype": settings.dtype,
           "lora": settings.lora.as_dict(), "sha256": hashes,
           "training": {"steps": settings.steps, "batch_size": settings.batch_size,
                        "lr": settings.lr, "seed": settings.seed}}
    write_json(out / CONFIG, cfg)
    write_json(out / "manifest.json", {
        "data_hash": result.data_hash, "n_train": result.n_train,
        "n_calibrate": result.n_calibrate, "n_test": result.n_test,
        "temperature": result.temperature, "metrics": result.metrics,
        "baseline": result.baseline, "losses": result.losses, "seconds": result.seconds,
        "settings": {k: str(v) for k, v in vars(settings).items()}})
    _write_card(out, cfg, result)


@dataclass
class Loaded:
    """A model ready to train: its tokenizer and tokenizer directory, the module, and the
    released checkpoint it continues from, if any."""

    tok: Any
    tok_dir: Path
    net: Any
    start: Source | None


def load_model(s: Settings, device: str) -> Loaded:
    """The tokenizer and the module for ``s``: a fresh adapter on the base, or the released
    checkpoint's adapter and head to continue from."""
    torch = _torch()
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    start = strands_source(s.init, download=s.download) if s.init else None
    if start is not None:
        base_dir, tok_dir = start.base_dir, start.tokenizer_dir
    elif isinstance(s.base, Path):
        base_dir = tok_dir = expand(s.base)
    else:
        paths = {p.filename: locate(p, download=s.download) for p in s.base.files}
        base_dir, tok_dir = paths["config.json"].parent, paths["tokenizer.json"].parent
    tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True, trust_remote_code=False)
    if start is not None:
        torso = load_torso(base_dir, start.lora_dir, s.dtype, device, trainable=True)
        state = load_file(str(start.head))
        head = build_head(torch, state["q.weight"].shape[1], start.pointer_dim)
        head.load_state_dict(state)
    else:
        torso = attach_features(load_torso(base_dir, None, s.dtype, device), s.lora)
        head = build_head(torch, torso.config.hidden_size, 256)
    return Loaded(tok, tok_dir, build_net(torch, torso, head.to(device).float()), start)


def fit_model(loaded: Loaded, cases: Sequence[Case], s: Settings, out: Path, device: str
              ) -> list[float]:
    """Train ``loaded.net`` for ``s.steps`` steps on ``cases``; returns the loss per step."""
    torch = _torch()
    net, tok = loaded.net, loaded.tok
    adapter = [p for _, p in net.torso.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": adapter, "scale": 1.0},
                                   {"params": list(net.head.parameters()),
                                    "scale": s.head_lr_scale}], lr=s.lr, weight_decay=0.0)

    def batch(step: int) -> dict[str, Any]:
        rng = random.Random(s.seed * 1_000_003 + step)  # noqa: S311
        rows, labels = [], []
        for c in (cases[rng.randrange(len(cases))] for _ in range(s.batch_size)):
            order = list(range(len(c.options)))
            rng.shuffle(order)
            rows.append(encode(tok, c, order, s.max_tokens))
            labels.append(order.index(c.label_index))
        return collate(torch, rows, labels, device)

    trainer = Trainer(net, optimizer, _loss, out=out / "run",
                      step=DeciderStep(net, optimizer, _loss, clip_grad_norm=1.0))
    report = trainer.fit(batch, steps=s.steps, schedule=warmup_cosine(
        s.lr, total_steps=s.steps, warmup_steps=max(1, s.steps // 10)), resume=False,
        write_checkpoints=False, log_every=1)
    shutil.copy(out / "run" / "metrics.jsonl", out / "train_log.jsonl")
    shutil.rmtree(out / "run", ignore_errors=True)
    return [h["loss"] for h in report.history]


def train(cases: Sequence[Case], out: Path | str, settings: Settings | None = None) -> Result:
    """Train on ``cases``, calibrate, score the test split and write the directory ``out``."""
    s = settings or Settings()
    out = expand(out)
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    fit_cases, cal_cases, test_cases = splits(cases, s)
    if min(len(fit_cases), len(cal_cases), len(test_cases)) < 1:
        raise DecideError("too few cases (or groups) to hold out calibration and test splits")
    _torch().manual_seed(s.seed)
    device = device_name(s.device)
    loaded = load_model(s, device)
    baseline = None
    if loaded.start is not None:
        baseline = metrics_of(predict(loaded.net, loaded.tok, test_cases, s, device),
                              test_cases, loaded.start.temperature)
    say(f"training {trainable_parameters(loaded.net)} parameters on {len(fit_cases)} cases "
        f"({len(cal_cases)} calibration, {len(test_cases)} test) on {device}")
    losses = fit_model(loaded, fit_cases, s, out, device)
    cal_logits = predict(loaded.net, loaded.tok, cal_cases, s, device)
    temperature = fit_temperature([softmax(r) for r in cal_logits],
                                  [c.label_index for c in cal_cases])
    test_logits = predict(loaded.net, loaded.tok, test_cases, s, device)
    result = Result(out, len(fit_cases), len(cal_cases), len(test_cases), temperature,
                    metrics_of(test_logits, test_cases, temperature), baseline, losses,
                    time.perf_counter() - began, fingerprint(list(cases)))
    _save(out, loaded.net, loaded.tok_dir, s, result)
    registry.register(out)
    return result


def _run(args: Namespace) -> int:
    cases = guard_cases() if args.cases == "guards" else read_cases(args.cases)
    init = STRANDS_V19 if args.init == "strands" else None
    settings = Settings(name=args.name, init=init, steps=args.steps, batch_size=args.batch_size,
                        lr=args.lr, seed=args.seed, device=args.device, download=args.download,
                        lora=Lora(args.rank, 2 * args.rank, 0.05, DEFAULT_TARGETS))
    try:
        got = train(cases, args.out, settings)
    except (DecideError, ValueError, OSError) as exc:
        warn(f"error: {exc}")
        return 2
    say(f"wrote {got.out} in {got.seconds:.0f}s: test accuracy {got.metrics['accuracy']:.3f}, "
        f"Brier {got.metrics['brier']:.3f}, ECE {got.metrics['ece']:.3f}, "
        f"temperature {got.temperature:.3f}")
    if got.baseline:
        say(f"before training: accuracy {got.baseline['accuracy']:.3f}, "
            f"Brier {got.baseline['brier']:.3f}, ECE {got.baseline['ece']:.3f}")
    if got.n_train < SMALL_DATA:
        say(f"{got.n_train} training cases: this checks the pipeline, it is not a quality claim")
    say(json.dumps({"data_hash": got.data_hash}))
    return 0


COMMANDS = Group(
    "ml-stack-train-decider",
    "Fine-tune a pointer-head decision model from labelled cases: a LoRA on a small base "
    "model and a pointer head, options shuffled on every pass, a temperature fitted on cases "
    "held apart from training and scoring. Writes a directory `PointerDecider` loads, with a "
    "model card that records the data hash, the splits and the metrics.",
    allow_abbrev=False, run=_run,
    options=[
        flag("cases", help="a JSONL file of labelled cases, or `guards`"),
        option("out", required=True), flag("--name", default="decider"),
        flag("--steps", type=int, default=60), flag("--batch-size", type=int, default=4),
        flag("--lr", type=float, default=2e-4), flag("--seed", type=int, default=0),
        flag("--rank", type=int, default=16),
        flag("--init", choices=("strands",), default=None,
             help="continue from the released checkpoint instead of a fresh adapter"),
        flag("--device", default="auto"),
        flag("--download", action="store_true", help="download the pinned base files"),
    ])


main = COMMANDS.run


if __name__ == "__main__":  # pragma: no cover - the entry point is `ml-stack-train-decider`
    raise SystemExit(main())
