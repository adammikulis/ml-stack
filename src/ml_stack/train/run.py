"""ml-stack-train-run: train one recipe from a config file.

With ``--lora`` the run is the whole path rather than only the training: what it will cost
is said before a weight is loaded and refused over the same 30-minute ceiling the bench
uses unless ``--yes``; the adapter is checkpointed instead of the frozen base; and
``--export-gguf`` merges it back in, converts through the managed llama.cpp build's own
source, and preflights the file the serve path would load. Every run that writes anything
writes a ``manifest.json`` naming its training data by hash and example count -- a
fine-tune whose data cannot be identified afterwards cannot be reproduced or believed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ml_stack.train.recipes import build, known, validate
from ml_stack.train.recipes.models import parameter_count
from ml_stack.train.schedule import warmup_cosine
from ml_stack.train.trainer import Trainer

ADAPTER_DIR = "adapter"
MERGED_DIR = "merged"


def _base_of(recipe_id: str, config: dict[str, Any], data: Path) -> tuple[str, dict[str, Any]]:
    """``(base, the size's contract entry)`` -- what a plan needs before anything loads.

    The same answer `build_tool_caller` reaches: the data's manifest names the base it was
    rendered for, and the recipe's size entry is both the fallback and where the parameter
    counts a wall-clock estimate needs are written down. A base the *data* names is not
    that size's model, so the size's counts are not about it and are not used for it.
    """
    from ml_stack.contracts import recipe

    spec = recipe(recipe_id)
    sizes = spec.get("sizes", {})
    if not sizes:
        return "", {}
    size = config.get("size") or sorted(sizes)[0]
    entry = dict(sizes.get(size, {}))
    manifest_base = ""
    manifest = Path(data).expanduser() / "manifest.json"
    if manifest.is_file():
        try:
            manifest_base = str(json.loads(manifest.read_text()).get("base") or "")
        except (OSError, json.JSONDecodeError):
            manifest_base = ""
    base = manifest_base or str(entry.get("base") or "")
    return base, ({} if manifest_base and manifest_base != entry.get("base") else entry)


def plan_for(recipe_id: str, config: dict[str, Any], data: Path, *,
             ceiling_min: float | None = None, seconds_per_step: float = 0.0) -> Any:
    """The `train.lora.Fit` for this run: what fits, and what it should take."""
    from ml_stack.train.lora import plan
    from ml_stack.train.recipes.tool_calls import device_for, read_conversations

    base, entry = _base_of(recipe_id, config, data)
    try:
        train, holdout, _ = read_conversations(data)
        examples = len(train) + len(holdout)
    except (OSError, ValueError):
        examples = 0
    return plan(config, base=base, device=str(device_for()), examples=examples,
                size_spec=entry, ceiling_min=ceiling_min,
                seconds_per_step=seconds_per_step)


def run(recipe_id: str, config: dict[str, Any], data: Path, out: Path,
        *, dry: bool = False, on_step: Any = None, export: bool = False,
        merge: bool = False, quant: str = "Q8_0", yes: bool = False,
        ceiling_min: float | None = None, say: Any = None) -> dict[str, Any]:
    from ml_stack.train.lora import fingerprint, refuse_over_ceiling

    config = validate(recipe_id, config)
    if dry:
        config = {**config, "steps": min(int(config.get("steps") or 20), 20)}
    talk = say or print
    lora = bool(config.get("lora"))
    out = Path(out).expanduser()

    fit = None
    if lora:
        fit = plan_for(recipe_id, config, data, ceiling_min=ceiling_min)
        for line in fit.lines():
            talk(line)
        # A dry run is 20 steps and is never refused: it is how the estimate above becomes
        # a measurement, the same way a smoke bench run is never refused.
        if not dry:
            refuse_over_ceiling(fit, yes=yes)

    built = build(recipe_id, config, data)
    steps = int(config["steps"])
    trainer = Trainer(built.model, built.optimizer, built.loss, out=out, step=built.step)
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
    result = {
        "recipe": recipe_id,
        "steps": report.steps,
        "final_loss": report.final_loss,
        "best_metric": report.best_metric,
        "checkpoint": str(report.last_checkpoint or ""),
        "parameters": parameter_count(built.model),
        "seconds": round(report.seconds, 1),
        "dry_run": dry,
    }

    if lora:
        result["lora"] = _finish_lora(built, config, data, out, report=report, fit=fit,
                                      dry=dry, export=export, merge=merge, quant=quant,
                                      talk=talk)
    if dry:
        return result
    if not lora:
        (out / "manifest.json").write_text(json.dumps(
            {"recipe": recipe_id, "base": built.config.get("base", ""), "config": config,
             "data": fingerprint(data),
             **{k: v for k, v in result.items() if k != "recipe"}}, indent=2, default=str))
    result["manifest"] = str(out / "manifest.json")
    return result


def _finish_lora(built: Any, config: dict[str, Any], data: Path, out: Path, *, report: Any,
                 fit: Any, dry: bool, export: bool, merge: bool, quant: str,
                 talk: Any) -> dict[str, Any]:
    """Adapter, merge, GGUF, preflight, manifest -- everything after the last step."""
    from ml_stack.train import lora as lora_mod

    base = str(built.config.get("base") or "")
    measured = report.seconds / report.steps if report.steps else 0.0
    said = fit
    if measured:
        said = lora_mod.plan(config, base=base,
                             device=str(built.config.get("device") or ""),
                             examples=int(built.config.get("rows") or 0),
                             trainable=int(built.config.get("trainable_parameters") or 0),
                             seconds_per_step=measured, ceiling_min=fit.ceiling_min)
        talk(f"measured: {measured:.1f} s/step over {report.steps} steps"
             + (f" -- {lora_mod.span(measured * fit.steps)} for the {fit.steps} steps this "
                "config asks for" if dry else ""))

    got: dict[str, Any] = {"settings": lora_mod.Lora.of(config).as_dict(),
                           "trainable_parameters": int(
                               built.config.get("trainable_parameters") or 0),
                           "seconds_per_step": round(measured, 3),
                           "plan": said.as_dict()}
    if dry:
        got["adapter"] = ""
        return got

    adapter = lora_mod.save_adapter(built.model, out / ADAPTER_DIR)
    talk(f"adapter: {adapter}")
    got["adapter"] = str(adapter)

    if export or merge:
        merged = lora_mod.merge(base, adapter, out / MERGED_DIR)
        talk(f"merged: {merged}")
        got["merged"] = str(merged)

    if export:
        result = lora_mod.export_gguf(out / MERGED_DIR, out,
                                      name=f"{Path(base).name}-tools", quant=quant)
        talk(f"export: {result.path} ({result.size_mb:.0f} MB)")
        got["gguf"] = str(result.path)
        got["gguf_sha256"] = result.sha256
        try:
            checked = lora_mod.preflight_export(result.path)
            got["preflight"] = lora_mod.summarise(checked)
            got["preflight_ok"] = bool(checked.ok)
        except Exception as exc:                      # noqa: BLE001 - no build to ask
            got["preflight"] = f"not checked: {exc}"
            got["preflight_ok"] = None
        talk(f"preflight: {got['preflight']}")

    manifest = {"recipe": "tool-calls", "base": base, "config": config,
                "data": lora_mod.fingerprint(data), "lora": got,
                "steps": report.steps, "final_loss": report.final_loss,
                "best_metric": report.best_metric, "seconds": round(report.seconds, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    talk(f"manifest: {out / 'manifest.json'}")
    return got


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="ml-stack-train-run")
    ap.add_argument("--recipe", required=True, choices=known())
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="", help="JSON file of settings")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--size", default="",
                    help="which size of the recipe: its base model, and the defaults that "
                         "suit it (tool-calls: 270m, e4b)")
    ap.add_argument("--dry-run", action="store_true",
                    help="20 steps, no checkpoint: does this config work at all, and what "
                         "does a step really cost")
    lora = ap.add_argument_group(
        "lora", "train an adapter instead of every weight -- what makes an 8B base "
                "trainable on one machine. Needs peft: pip install 'ml-stack[train-lora]'")
    lora.add_argument("--lora", action="store_true", help="train a LoRA adapter")
    lora.add_argument("--lora-rank", type=int, default=None)
    lora.add_argument("--lora-alpha", type=int, default=None)
    lora.add_argument("--lora-dropout", type=float, default=None)
    lora.add_argument("--lora-targets", default="", metavar="a,b,c",
                      help="which projections get an adapter; empty means attention and "
                           "MLP both")
    lora.add_argument("--merge", action="store_true",
                      help="fold the adapter back into the base, in Hugging Face layout")
    lora.add_argument("--export-gguf", action="store_true",
                      help="merge, then convert and quantise into --out, then preflight it")
    lora.add_argument("--quant", default="Q8_0", help="the GGUF quantisation to end with")
    lora.add_argument("--ceiling", type=float, default=None, metavar="MINUTES",
                      help="refuse a run estimated to take longer than this (default 30)")
    lora.add_argument("--yes", action="store_true", help="run past the ceiling")
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
    if a.size:
        config["size"] = a.size
    if a.lora:
        config["lora"] = True
    for name in ("lora_rank", "lora_alpha", "lora_dropout"):
        value = getattr(a, name)
        if value is not None:
            config[name] = value
    if a.lora_targets:
        config["lora_targets"] = a.lora_targets

    from ml_stack.train.lora import OverCeiling

    try:
        result = run(a.recipe, config, Path(a.data), Path(a.out), dry=a.dry_run,
                     export=a.export_gguf, merge=a.merge, quant=a.quant, yes=a.yes,
                     ceiling_min=a.ceiling)
    except OverCeiling as exc:
        print(str(exc), file=sys.stderr)
        return 5
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:                       # ToolNotFound, ConversionError
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
