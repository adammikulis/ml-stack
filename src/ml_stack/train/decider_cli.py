"""``ml-stack-train-decider`` and ``ml-stack-decide train``: the command line of
`ml_stack.train.decider`."""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path

from ml_stack.command import Group, flag, option
from ml_stack.decide import dataset, metrics, registry
from ml_stack.decide.cases import Case
from ml_stack.decide.guards import guard_cases
from ml_stack.decide.pins import BASES, QWEN35_0_8B_BASE, STRANDS_V19, Checkpoint
from ml_stack.decide.types import DecideError
from ml_stack.home import expand
from ml_stack.log import say, warn
from ml_stack.train.decider import Settings, check_inputs, split_cases, train
from ml_stack.train.lora import DEFAULT_TARGETS, Lora

__all__ = ["COMMANDS", "OPTIONS", "main", "run"]

def _base(value: str) -> Checkpoint | Path:
    if value in BASES:
        return BASES[value]
    path = expand(value)
    if not (path / "config.json").is_file():
        raise DecideError(f"{value!r} is neither a pinned base ({sorted(BASES)}) nor a "
                          "directory holding a model's config.json")
    return path


def _settings(args: Namespace) -> Settings:
    return Settings(
        name=args.name, base=_base(args.base), init=STRANDS_V19 if args.init == "strands" else None,
        steps=args.steps, batch_size=args.batch_size, max_tokens=args.max_tokens, lr=args.lr, seed=args.seed,
        device=args.device, download=args.download, baseline=args.baseline,
        allow_worse=args.allow_worse, replace=args.replace, wait_s=args.wait,
        floor=args.floor, allow_repo=args.allow_in_repo, lora=Lora(args.rank, 2 * args.rank, 0.05, DEFAULT_TARGETS))


def _show(s: Settings, cases: Sequence[Case], eval_cases: Sequence[Case] | None, out: Path
          ) -> list[str]:
    found = dataset.check(cases, label="training data")
    say(f"{len(cases)} cases, labels {dict(sorted(found.labels.items()))}, kinds "
        f"{dict(sorted(found.kinds.items()))}")
    warnings = check_inputs(cases, eval_cases, s, out)
    fit, cal, test = split_cases(cases, eval_cases, s)
    say(f"splits: {len(fit)} train, {len(cal)} calibration, {len(test)} test"
        + (" (the evaluation file)" if eval_cases is not None else " (whole groups)"))
    return warnings


def run(args: Namespace) -> int:
    try:
        s = _settings(args)
        registry.check_name(s.name)
        cases = guard_cases() if args.data == "guards" else dataset.load(args.data)
        eval_cases = dataset.load(args.eval) if args.eval else None
        out = expand(args.out) if args.out else registry.models_dir() / s.name
        if args.dry_run:
            for w in _show(s, cases, eval_cases, out):
                warn(w)
            say(f"dry run: would write {out}; nothing was trained or registered")
            return 0
        got = train(cases, out, s, eval_cases=eval_cases)
    except (DecideError, ValueError, OSError) as exc:
        warn(f"error: {exc}")
        return 2
    m = got.metrics
    say(f"wrote {got.out} in {got.seconds:.0f}s: test accuracy {m['accuracy']:.3f}, "
        f"Brier {m['brier']:.3f}, ECE {m['ece']:.3f}, abstain rate {m['abstain_rate']:.3f} at "
        f"{m['floor']:g}, temperature {got.temperature:.3f}")
    if got.baseline:
        b = got.baseline
        say(f"baseline ({got.baseline_name}): accuracy {b['accuracy']:.3f}, "
            f"Brier {b['brier']:.3f}, ECE {b['ece']:.3f}")
    for w in got.warnings:
        warn(w)
    say(f"registered as {s.name!r}; score it with: ml-stack-decide eval FILE --decider {s.name}")
    say(json.dumps({"data_hash": got.data_hash}))
    return 0


OPTIONS = [
    flag("--data", required=True, help="a JSONL file of labelled cases, or `guards`"),
    flag("--name", required=True, help="the name the decider is registered under"),
    flag("--eval", default="", help="a JSONL file of held-out cases to score on; refused if "
                                    "it shares a case or group with --data"),
    flag("--base", default=QWEN35_0_8B_BASE.name,
         help=f"a pinned base ({', '.join(sorted(BASES))}) or a local model directory"),
    flag("--baseline", default="auto",
         help="what the result must not be worse than: auto, majority, strands, none or a "
              "registered decider"),
    flag("--allow-worse", action="store_true",
         help="register even if worse than the baseline on the test cases"),
    flag("--replace", action="store_true",
         help="replace a decider already registered under --name (it is kept as NAME.prev)"),
    flag("--floor", type=float, default=metrics.FLOOR,
         help="the confidence under which an answer counts as abstained, for the abstain rate"),
    flag("--wait", type=float, default=0.0,
         help="seconds to wait for another training run to release the GPU"),
    option("out", help="where to write it (default: under the state directory, never a git "
                       "work tree)"),
    flag("--allow-in-repo", action="store_true",
         help="allow --out inside a git work tree"),
    option("dry-run"),
    flag("--steps", type=int, default=60), flag("--batch-size", type=int, default=4),
    flag("--lr", type=float, default=2e-4), flag("--seed", type=int, default=0),
    flag("--rank", type=int, default=16),
    flag("--max-tokens", type=int, default=1024,
         help="the longest prompt a case may render to; longer cases are refused"),
    flag("--init", choices=("strands",), default=None,
         help="continue from the released checkpoint instead of a fresh adapter"),
    flag("--device", default="auto"),
    flag("--download", action="store_true", help="download the pinned base files"),
]

COMMANDS = Group(
    "ml-stack-train-decider",
    "Fine-tune a pointer-head decision model from labelled cases: a LoRA on a small base "
    "model and a pointer head, options shuffled on every pass, a temperature per question "
    "kind fitted on cases held apart from training and scoring. Checks the data first, holds "
    "the GPU at the Broker, registers the result unless it is worse than its baseline. "
    "Writes a directory `PointerDecider` loads, with a model card that records the data "
    "hash, the splits and the metrics.",
    allow_abbrev=False, run=run, options=OPTIONS)


main = COMMANDS.run


if __name__ == "__main__":  # pragma: no cover - the entry point is `ml-stack-train-decider`
    raise SystemExit(main())
