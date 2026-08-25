"""ml-stack-train-run: train one recipe from a config file."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ml_stack.train.recipes import build, known, validate
from ml_stack.train.recipes.models import parameter_count
from ml_stack.train.schedule import warmup_cosine
from ml_stack.train.trainer import Trainer


def run(recipe_id: str, config: dict[str, Any], data: Path, out: Path,
        *, dry: bool = False, on_step: Any = None) -> dict[str, Any]:
    config = validate(recipe_id, config)
    if dry:
        config = {**config, "steps": min(int(config.get("steps") or 20), 20)}

    built = build(recipe_id, config, data)
    steps = int(config["steps"])
    trainer = Trainer(built.model, built.optimizer, built.loss, out=out)
    report = trainer.fit(
        built.batches, steps=steps,
        schedule=warmup_cosine(float(config["learning_rate"]), total_steps=steps,
                               warmup_steps=max(1, steps // 20)),
        eval_data=built.eval_batches,
        eval_every=int(config.get("eval_every") or max(1, steps // 10)),
        checkpoint_every=0 if dry else int(config.get("checkpoint_every")
                                           or max(1, steps // 5)),
        config={**built.config, "parameters": parameter_count(built.model)},
        write_checkpoints=not dry,
        on_step=on_step,
    )
    return {
        "recipe": recipe_id,
        "steps": report.steps,
        "final_loss": report.final_loss,
        "best_metric": report.best_metric,
        "checkpoint": str(report.last_checkpoint or ""),
        "parameters": parameter_count(built.model),
        "seconds": round(report.seconds, 1),
        "dry_run": dry,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="ml-stack-train-run")
    ap.add_argument("--recipe", required=True, choices=known())
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="", help="JSON file of settings")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--dry-run", action="store_true",
                    help="20 steps, no checkpoint: does this config work at all")
    a = ap.parse_args(argv)

    config: dict[str, Any] = {}
    if a.config:
        config = json.loads(Path(a.config).expanduser().read_text())
    for pair in a.set:
        key, _, value = pair.partition("=")
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            config[key] = value

    try:
        result = run(a.recipe, config, Path(a.data), Path(a.out), dry=a.dry_run)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
