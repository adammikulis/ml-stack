"""``poolhouse-train-decider``: fine-tune a pointer-head decision model from labelled cases.

    poolhouse-train-decider guards --out ./my-decider --steps 200
    poolhouse-train-decider cases.jsonl --out ./my-decider --init strands --download

A LoRA on a small base model plus a ~1M-parameter pointer head, trained on the cases with
the options shuffled on every pass, then a temperature per question kind fitted on cases
held apart from both training and scoring. The data is checked first, the run holds the GPU
at the Broker, and a decider worse than its baseline on the test cases is not registered.
The output directory holds ``head.safetensors``, ``lora/``, the
tokenizer files, ``decider.json`` (base pins, file hashes, temperature), ``manifest.json``
(data hash, split sizes, metrics) and ``model_card.md``, and `PointerDecider` loads it.
"""

from __future__ import annotations

import random
import shutil
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poolhouse import sentinel
from poolhouse.decide import dataset, metrics, pointer_prompt, registry
from poolhouse.decide.base import text_of
from poolhouse.decide.calibrate import fit_temperature, softmax
from poolhouse.decide.cases import Case, fingerprint
from poolhouse.decide.fetch import locate
from poolhouse.decide.pins import QWEN35_0_8B_BASE, STRANDS_V19, Checkpoint
from poolhouse.decide.pointer import PointerDecider, build_head, device_name, load_torso
from poolhouse.decide.sources import CONFIG, FORMAT, Source, question_key, strands_source
from poolhouse.decide.types import DecideError
from poolhouse.files import sha256_file, write_json
from poolhouse.home import expand
from poolhouse.log import say
from poolhouse.train import gpu
from poolhouse.train.holdout import by_group
from poolhouse.train.lora import (
    DEFAULT_TARGETS,
    Lora,
    require_peft,
    trainable_parameters,
)
from poolhouse.train.schedule import warmup_cosine
from poolhouse.train.step import TorchStep
from poolhouse.train.trainer import Trainer

__all__ = ["Result", "Settings", "check_inputs", "train"]

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
    baseline: str = "auto"
    """``auto`` (the checkpoint continued from, else the majority label), ``majority``,
    ``strands``, ``none`` or the name of a registered decider."""
    allow_worse: bool = False
    replace: bool = False
    wait_s: float = 0.0
    floor: float = metrics.FLOOR
    allow_repo: bool = False


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
    temperatures: dict[str, float] = field(default_factory=dict)
    baseline_name: str = ""
    warnings: list[str] = field(default_factory=list)
    refusal: list[str] = field(default_factory=list)
    registered: bool = False
    kinds: dict[str, str] = field(default_factory=dict)


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


def metrics_of(logits: Sequence[Sequence[float]], cases: Sequence[Case], temperature: float,
               by_kind: dict[str, float] | None = None, floor: float = metrics.FLOOR
               ) -> dict[str, float]:
    """`metrics.Metrics` (as a dict) of ``logits`` scaled by ``temperature``, or by the
    temperature of each case's kind in ``by_kind``."""
    temps = by_kind or {}
    rows = [softmax([x / temps.get(c.kind_name, temperature) for x in r])
            for r, c in zip(logits, cases, strict=True)]
    return metrics.of_rows(rows, [c.label_index for c in cases], floor=floor).public()


def splits(cases: Sequence[Case], settings: Settings) -> tuple[list[Case], list[Case], list[Case]]:
    """Train, calibration and test cases, split by whole groups so no group is in two. A case
    without a group is grouped with the cases that say the same thing."""
    groups = [c.group or dataset.key_of(c) for c in cases]
    rest = by_group(cases, groups, settings.test_fraction, seed=settings.seed)
    inner = by_group(rest.train, [c.group or dataset.key_of(c) for c in rest.train],
                     settings.calibrate_fraction / (1 - settings.test_fraction),
                     seed=settings.seed + 1)
    return list(inner.train), list(inner.holdout), list(rest.holdout)


MIN_KIND_CASES = 20
ECE_FIT_CASES = 50


def fit_temperatures(logits: Sequence[Sequence[float]], cases: Sequence[Case]
                     ) -> tuple[float, dict[str, float]]:
    """The temperature over every calibration case, and one for each kind with at least
    `MIN_KIND_CASES` of them. A set of `ECE_FIT_CASES` or more is fitted to minimise ECE,
    a smaller one to minimise NLL."""
    def fit_on(idx: list[int]) -> float:
        rows = [softmax(logits[i]) for i in idx]
        labels = [cases[i].label_index for i in idx]
        return (metrics.fit_temperature_ece if len(idx) >= ECE_FIT_CASES else fit_temperature)(rows, labels)

    everything = fit_on(list(range(len(cases))))
    by_kind: dict[str, float] = {}
    for kind in sorted({c.kind_name for c in cases}):
        idx = [i for i, c in enumerate(cases) if c.kind_name == kind]
        if len(idx) >= MIN_KIND_CASES:
            by_kind[kind] = fit_on(idx)
    return everything, by_kind


def question_kinds(cases: Sequence[Case]) -> dict[str, str]:
    """Each question text that always has one kind, mapped to it, so the decider can pick the
    temperature of a question it is asked."""
    seen: dict[str, set[str]] = {}
    for c in cases:
        seen.setdefault(question_key(c.question), set()).add(c.kind_name)
    return {q: next(iter(k)) for q, k in sorted(seen.items()) if len(k) == 1}


def majority_rows(fit_cases: Sequence[Case], cases: Sequence[Case]) -> list[list[float]]:
    """Probabilities from the training label frequencies alone: the floor any model must beat."""
    counts: dict[str, int] = {}
    for c in fit_cases:
        counts[c.label] = counts.get(c.label, 0) + 1
    out = []
    for c in cases:
        raw = [counts.get(o.name, 0) + 1.0 for o in c.options]
        out.append([x / sum(raw) for x in raw])
    return out


def baseline_of(spec: str, fit_cases: Sequence[Case], test: Sequence[Case], s: Settings
                ) -> tuple[str, dict[str, float] | None]:
    """The name and test metrics of the baseline ``spec`` names, scored on ``test``."""
    if spec == "none":
        return "", None
    if spec == "majority":
        rows = majority_rows(fit_cases, test)
        return "majority label", metrics.of_rows(
            rows, [c.label_index for c in test], floor=s.floor).public()
    source: Checkpoint | Path = STRANDS_V19 if spec == "strands" else registry.find(spec)
    decider = PointerDecider(source, device=s.device, download=s.download)
    rows = []
    for c in test:
        got = decider.decide(c.question, c.state, c.options)
        rows.append([got.scores[o.name] for o in c.options])
    return spec, metrics.of_rows(rows, [c.label_index for c in test], floor=s.floor).public()


def _safe(text: str, limit: int = 48) -> str:
    """``text`` cut to ``limit`` characters and reduced to ones that cannot alter a Markdown
    card."""
    return "".join(ch if ch.isalnum() or ch in " _.-" else "?" for ch in text)[:limit]


def _row(label: str, m: dict[str, float]) -> str:
    return (f"| {label} | {m['accuracy']:.3f} | {m['brier']:.3f} | {m['ece']:.3f} | "
            f"{m['abstain_rate']:.3f} |")


def _write_card(out: Path, cfg: dict[str, Any], result: Result) -> None:
    small = result.n_train < dataset.WARN_CASES
    m, b = result.metrics, result.baseline
    floor = m["floor"]
    lines = [f"# {_safe(cfg['name'])}", "",
             "A pointer-head decision model: a LoRA on "
             f"`{_safe(cfg['base']['repo'], 80)}` and a pointer head, trained by "
             "`poolhouse-decide train`.", ""]
    if small:
        lines += [f"**Pipeline check, not a quality claim.** Trained on {result.n_train} cases "
                  f"for {cfg['training']['steps']} steps.", ""]
    if result.refusal:
        lines += ["**Not registered:** " + "; ".join(result.refusal), ""]
    lines += ["## Data", f"- data hash (SHA-256 of the cases): `{result.data_hash}`",
              f"- cases: {result.n_train} train, {result.n_calibrate} calibration, "
              f"{result.n_test} test (whole groups; no group is in two)",
              "- the licence of the data is the licence of the cases you trained on"]
    lines += [f"- warning: {_safe(w, 200)}" for w in result.warnings]
    lines += ["", "## Results on the test cases",
              f"| | accuracy | Brier | ECE | abstain rate at {floor:g} |", "|---|---|---|---|---|"]
    if b:
        lines.append(_row(f"baseline: {_safe(result.baseline_name)}", b))
    lines += [_row("this decider", m), "",
              f"Accuracy among answered cases (confidence at least {floor:g}): "
              f"{m['answered_accuracy']:.3f}. Brier is the sum over options of the squared "
              "error, 0 to 2; ECE uses 10 bins.", "",
              "## Calibration",
              f"Temperature {result.temperature:.3f} over all calibration cases"
              + ("; by kind: " + ", ".join(f"{k} {t:.3f}" for k, t in
                                           sorted(result.temperatures.items()))
                 if result.temperatures else "") + ".", "",
              "## Files",
              "`head.safetensors`, `lora/adapter_model.safetensors`, `decider.json`, "
              "`manifest.json`. The base model is "
              + (f"the local directory {_safe(cfg['base']['path'], 200)}."
                 if "path" in cfg["base"] else
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
           "pointer_dim": 256, "temperature": result.temperature,
           "temperature_by_kind": result.temperatures, "question_kinds": result.kinds,
           "dtype": settings.dtype,
           "lora": settings.lora.as_dict(), "sha256": hashes,
           "training": {"steps": settings.steps, "batch_size": settings.batch_size,
                        "lr": settings.lr, "seed": settings.seed}}
    write_json(out / CONFIG, cfg)
    write_json(out / "manifest.json", {
        "data_hash": result.data_hash, "n_train": result.n_train,
        "n_calibrate": result.n_calibrate, "n_test": result.n_test,
        "temperature": result.temperature, "metrics": result.metrics,
        "baseline": result.baseline, "baseline_name": result.baseline_name,
        "warnings": result.warnings, "refusal": result.refusal, "registered": result.registered,
        "temperature_by_kind": result.temperatures,
        "losses": result.losses, "seconds": result.seconds,
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


def work_tree(path: Path) -> Path | None:
    """The nearest directory above ``path`` that holds a ``.git`` entry, else None."""
    for parent in (path.resolve(), *path.resolve().parents):
        if (parent / ".git").exists():
            return parent
    return None


PINNED = (("head.safetensors", "model"), ("lora/adapter_model.safetensors", "model"),
          ("lora/adapter_config.json", "config"), (CONFIG, "config"))


def pin_files(root: Path, name: str) -> None:
    """Pin the decider's weights and config in the sentinel, so a change after training is a
    finding. A failure part way unpins what was pinned."""
    node = sentinel.armed()
    if node.mode == sentinel.Mode.OFF:
        return
    done = False
    try:
        for rel, kind in PINNED:
            node.manifest.pin(root / rel, kind, source=f"trained:{name}")
        done = True
    finally:
        if not done:
            unpin_files(root)


def unpin_files(root: Path) -> None:
    """Drop the pins `pin_files` made for the decider in ``root``."""
    node = sentinel.default()
    for rel, _ in PINNED:
        node.manifest.unpin(root / rel)


def _gate(s: Settings, result: Result) -> list[str]:
    """Why the result may not be registered: worse than its baseline on the test cases."""
    if result.baseline is None:
        return []
    ours = metrics.Metrics(**result.metrics)
    return metrics.worse_than(ours, metrics.Metrics(**result.baseline))


def check_inputs(cases: Sequence[Case], eval_cases: Sequence[Case] | None, s: Settings,
                 out: Path) -> list[str]:
    """Refuse before any work: a bad name, a name already registered, an output directory in
    use, a bad training or evaluation set, or an evaluation set that overlaps training.
    Returns the warnings."""
    registry.check_name(s.name)
    if not s.allow_repo and (tree := work_tree(out)) is not None:
        raise DecideError(f"{out} is inside the git work tree {tree}: trained weights and data "
                          "are not committed. Write to the state directory (the default), or "
                          "pass --allow-in-repo")
    if registry.taken(s.name) and not s.replace:
        raise DecideError(f"a decider called {s.name!r} is already registered; pass replace to "
                          "register this run instead (the old one is kept as "
                          f"{s.name + registry.PREVIOUS!r})")
    if out.exists() and any(out.iterdir()):
        raise DecideError(f"{out} is not empty; a run never writes into an existing directory")
    found = dataset.check(cases, label="training data")
    warnings = list(found.warnings)
    problems = list(found.errors)
    if eval_cases is not None:
        held = dataset.check(eval_cases, label="evaluation data")
        problems += [e for e in held.errors
                     if "needed to hold out" not in e and "nothing to learn" not in e]
        warnings += held.warnings + [e for e in held.errors if "nothing to learn" in e]
        problems += dataset.leaks(cases, eval_cases)
    if problems:
        raise dataset.DataError("; ".join(problems))
    return warnings


def split_cases(cases: Sequence[Case], eval_cases: Sequence[Case] | None, s: Settings
           ) -> tuple[list[Case], list[Case], list[Case]]:
    if eval_cases is None:
        return splits(cases, s)
    inner = by_group(list(cases), [c.group or dataset.key_of(c) for c in cases],
                     s.calibrate_fraction, seed=s.seed)
    return list(inner.train), list(inner.holdout), list(eval_cases)


def train(cases: Sequence[Case], out: Path | str, settings: Settings | None = None, *,
          eval_cases: Sequence[Case] | None = None) -> Result:
    """Train on ``cases``, calibrate, score the test split and write the directory ``out``.

    The test cases are ``eval_cases`` when given (refused if they share a case or group with
    ``cases``), else a share of whole groups. The decider is registered unless it is worse
    than its baseline on them and ``settings.allow_worse`` is not set, in which case the
    directory is written and `DecideError` names why.
    """
    s = settings or Settings()
    out = expand(out)
    warnings = check_inputs(cases, eval_cases, s, out)
    fit_cases, cal_cases, test_cases = split_cases(cases, eval_cases, s)
    if min(len(fit_cases), len(cal_cases), len(test_cases)) < 1:
        raise DecideError("too few cases (or groups) to hold out calibration and test splits")
    if overlap := dataset.leaks([*fit_cases, *cal_cases], test_cases, held_label="test"):
        raise dataset.DataError("; ".join(overlap))
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    spec = ("strands" if s.init else "majority") if s.baseline == "auto" else s.baseline
    device = device_name(s.device)
    hold = (nullcontext() if device == "cpu"
            else gpu.hold(f"train decider {s.name}", wait_s=s.wait_s))
    with hold:
        base_name, base = baseline_of(spec, fit_cases, test_cases, s)
        _torch().manual_seed(s.seed)
        loaded = load_model(s, device)
        say(f"training {trainable_parameters(loaded.net)} parameters on {len(fit_cases)} cases "
            f"({len(cal_cases)} calibration, {len(test_cases)} test) on {device}")
        losses = fit_model(loaded, fit_cases, s, out, device)
        cal_logits = predict(loaded.net, loaded.tok, cal_cases, s, device)
        temperature, by_kind = fit_temperatures(cal_logits, cal_cases)
        test_logits = predict(loaded.net, loaded.tok, test_cases, s, device)
        result = Result(out, len(fit_cases), len(cal_cases), len(test_cases), temperature,
                        metrics_of(test_logits, test_cases, temperature, by_kind, s.floor), base,
                        losses, time.perf_counter() - began, fingerprint(list(cases)),
                        by_kind, base_name, warnings)
        result.refusal = _gate(s, result)
        result.registered = not result.refusal or s.allow_worse
        result.kinds = question_kinds(cases)
        _save(out, loaded.net, loaded.tok_dir, s, result)
    if result.registered:
        registry.register(out, replace=s.replace)
        pin_files(out, s.name)
    else:
        raise DecideError(f"{s.name!r} is worse than the baseline ({base_name}) on the "
                          f"{result.n_test} test cases: {'; '.join(result.refusal)}. The "
                          f"directory {out} is written and not registered; allow_worse "
                          "registers it anyway")
    return result
