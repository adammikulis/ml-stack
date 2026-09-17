"""ml-stack-train-run: train one recipe from a config file.

Every run that writes anything writes a ``manifest.json`` naming its training data by hash
and example count. ``--lora`` trains an adapter, plans its cost before a weight loads, and
``--export-gguf`` merges, converts and preflights it. ``--detach`` re-runs the command in
its own session, recorded through `ml_stack.jobs` as the ``train`` job, which ``wait``,
``stop`` and ``status`` read back; a second one beside a live run is refused.
"""

from __future__ import annotations

import importlib
import json
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ml_stack import jobs
from ml_stack.files import versioned, write_json
from ml_stack.home import state
from ml_stack.log import say, warn
from ml_stack.train import lora
from ml_stack.train.metrics import MetricsLog
from ml_stack.train.probes import probe_hook
from ml_stack.train.recipes import Built, build, known, spec, tool_calls, validate
from ml_stack.train.recipes.models import parameter_count
from ml_stack.train.schedule import constant, warmup_cosine
from ml_stack.train.trainer import Trainer, TrainReport

#: 1 -- the recipe, the base, the config, the data fingerprint and the run's numbers.
MANIFEST_VERSION = 1

ADAPTER_DIR = "adapter"
MERGED_DIR = "merged"
EXPORT_DIR = "export"
FINISHED = ("completed", "already complete")
"""The stop reasons after which a run's export is written."""

KIND = "train"
"""The kind of job a detached run is recorded as, in `ml_stack.jobs`."""

WORDS = ("wait", "stop", "status", "parity")
"""What the command does instead of training, when one is written where the flags go."""


def home_dir() -> Path:
    """Where a detached run's log and its record of itself live, not its checkpoints."""
    return state("train")


def _jobs_home(home: Path | None = None) -> Path:
    """The `ml_stack.jobs` record directory under a training home."""
    return Path(home) / "jobs" if home is not None else home_dir() / "jobs"


def running(*, home: Path | None = None) -> int:
    """The pid of the detached training run, or 0 when none is alive."""
    return jobs.alive(KIND, home=_jobs_home(home))


def current(*, home: Path | None = None) -> dict[str, Any]:
    """The live detached training run's record -- pid, argv, log, started -- or ``{}``."""
    return jobs.recorded(KIND, home=_jobs_home(home)) if running(home=home) else {}


def detach(argv: Sequence[str]) -> Path:
    """Run ``ml-stack-train-run argv`` owned by no terminal, and return the log it writes."""
    rest = [a for a in argv if a != "--detach"]
    log = home_dir() / "logs" / f"train-{time.strftime('%Y%m%dT%H%M%S')}.log"
    return jobs.detach("ml_stack.train.run", rest, log=log, kind=KIND, home=_jobs_home()).log


def wait(*, say: Callable[[str], None] = say, home: Path | None = None,
         every: float = 60.0) -> int:
    """``ml-stack-train-run wait``: block until the detached run this machine records has
    ended, saying so every minute -- so what follows it is `wait && next` rather than a
    loop written by hand."""
    return jobs.wait(KIND, say=say, every=every, home=_jobs_home(home))


def stop(*, say: Callable[[str], None] = say, home: Path | None = None) -> int:
    """``ml-stack-train-run stop``: end the detached run. Whatever it checkpointed stays
    where ``--out`` put it."""
    return jobs.stop(KIND, say=say, home=_jobs_home(home))


def status(*, say: Callable[[str], None] = say, home: Path | None = None) -> int:
    """``ml-stack-train-run status``: what this machine records -- the training run and any
    other long command, the same lines `ml-stack-jobs status` prints."""
    return jobs.status(say=say, home=_jobs_home(home))


def parity(*, say: Callable[[str], None] = say, first: str = "torch",
           second: str = "mlx") -> int:
    """``ml-stack-train-run parity``: every array operation on two backends, side by side.

    Prints one row per operation with the largest absolute difference between the two,
    and returns 1 if any of them disagree by more than the tolerance.
    """
    from ml_stack.train.parity import report

    return report(say=say, first=first, second=second)


def _base_of(recipe_id: str, config: dict[str, Any], data: Path) -> tuple[str, dict[str, Any]]:
    """``(base, the size's contract entry)`` -- what a plan needs before anything loads.

    The same answer `build_tool_caller` reaches: the data's manifest names the base it was
    rendered for, and the recipe's size entry is both the fallback and where the parameter
    counts a wall-clock estimate needs are written down. A base the *data* names is not
    that size's model, so the size's counts are not about it and are not used for it.
    """
    sizes = spec(recipe_id).get("sizes", {})
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

    base, entry = _base_of(recipe_id, config, data)
    try:
        train, holdout, _ = tool_calls.read_conversations(data)
        examples = len(train) + len(holdout)
    except (OSError, ValueError):
        examples = 0
    return lora.plan(config, base=base, device=str(tool_calls.device_for()), examples=examples,
                size_spec=entry, ceiling_min=ceiling_min,
                seconds_per_step=seconds_per_step)


def run(recipe_id: str, config: dict[str, Any], data: Path, out: Path,
        *, dry: bool = False, on_step: Any = None, export: bool = False,
        merge: bool = False, quant: str = "Q8_0", yes: bool = False,
        ceiling_min: float | None = None, say: Any = None,
        should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Train ``recipe_id`` into ``out``, then its phases and its export; returns the result."""
    config = validate(recipe_id, config)
    if dry:
        config = {**config, "steps": min(int(config.get("steps") or 20), 20)}
    talk = say or print
    wants_lora = bool(config.get("lora"))
    out = Path(out).expanduser()

    fit = None
    if wants_lora:
        fit = plan_for(recipe_id, config, data, ceiling_min=ceiling_min)
        for line in fit.lines():
            talk(line)
        # A dry run is 20 steps and is never refused: it is how the estimate above becomes
        # a measurement, the same way a smoke bench run is never refused.
        if not dry:
            lora.refuse_over_ceiling(fit, yes=yes)

    built = build(recipe_id, config, data, framework=config["framework"])
    if built.batches is None:
        raise ValueError(f"{recipe_id} built no batches from {data}")
    steps = int(config["steps"])
    eval_every = int(config.get("eval_every") or max(1, steps // 10))
    hooks = [*built.hooks,
             *([probe_hook(out, built.predict, eval_every)] if built.predict else [])]
    rounds = Rounds(built, out, recipe_id, float(config["max_minutes"]) * 60, {
        "eval_data": built.eval_batches, "eval_every": eval_every,
        "checkpoint_every": 0 if dry else int(config.get("checkpoint_every")
                                              or max(1, steps // 5)),
        "write_checkpoints": not dry, "on_step": on_step, "should_stop": should_stop,
        "hooks": hooks})
    report = rounds.fit(steps, warmup_cosine(float(config["learning_rate"]),
                                             total_steps=steps,
                                             warmup_steps=max(1, steps // 20)))
    if not dry and report.stop_reason == "completed":
        report = rounds.phases(report)
    result = {
        "recipe": recipe_id,
        "steps": report.steps,
        "final_loss": report.final_loss,
        "best_metric": rounds.best,
        "checkpoint": str(report.last_checkpoint or ""),
        "parameters": parameter_count(built.model),
        "seconds": round(time.monotonic() - rounds.started, 1),
        "dry_run": dry,
        "stop_reason": report.stop_reason,
        "phases": rounds.done,
    }
    if not dry and built.export is not None and report.stop_reason in FINISHED:
        (out / EXPORT_DIR).mkdir(parents=True, exist_ok=True)
        result["export"] = built.export(out / EXPORT_DIR)

    if wants_lora:
        result["lora"] = _finish_lora(built, config, data, out, report=report, fit=fit,
                                      dry=dry, export=export, merge=merge, quant=quant,
                                      talk=talk)
    if dry:
        return result
    if not wants_lora:
        write_json(out / "manifest.json", versioned(
            {"recipe": recipe_id, "base": built.config.get("base", ""), "config": config,
             "data": lora.fingerprint(data),
             **{k: v for k, v in result.items() if k != "recipe"}}, MANIFEST_VERSION),
            default=str)
    result["manifest"] = str(out / "manifest.json")
    return result


class Rounds:
    """The main fit and each phase after it, over one model, one ``out`` and one time budget."""

    def __init__(self, built: Built, out: Path, recipe_id: str, budget_s: float,
                 common: dict[str, Any]) -> None:
        self.built, self.out, self.recipe_id = built, out, recipe_id
        self.budget_s, self.common = budget_s, common
        self.started = time.monotonic()
        self.best: float | None = None
        self.done: list[dict[str, Any]] = []

    def left(self) -> float:
        """Seconds of the budget still unspent; 0 means no limit."""
        if not self.budget_s:
            return 0.0
        return max(1e-9, self.budget_s - (time.monotonic() - self.started))

    def fit(self, steps: int, schedule: Any, continue_from: int | None = None) -> TrainReport:
        built = self.built
        trainer = Trainer(built.model, built.optimizer, built.loss, out=self.out,
                          step=built.step)
        report = trainer.fit(
            built.batches, steps=steps, schedule=schedule, continue_from=continue_from,
            max_seconds=self.left(), **self.common,
            config={**built.config, "recipe": self.recipe_id,
                    "framework": trainer.framework,
                    "parameters": parameter_count(built.model)})
        if report.best_metric is not None:
            self.best = min(report.best_metric, self.best if self.best is not None
                            else report.best_metric)
        return report

    def phases(self, report: TrainReport) -> TrainReport:
        """Each phase the recipe declares, in order, until one prepares nothing or ends early."""
        for phase in self.built.phases:
            if self.budget_s and time.monotonic() - self.started >= self.budget_s:
                report.stop_reason = "time_limit"
                break
            said = phase.prepare()
            if said is None:
                break
            before = report.steps
            with MetricsLog(self.out / "metrics.jsonl", resume=True) as log:
                log.note("phase", step=before, phase=phase.name, prepared=said)
            report = self.fit(before + phase.steps, constant(phase.learning_rate),
                              continue_from=before)
            self.done.append({"name": phase.name, "prepared": said,
                              "steps": report.steps - before})
            if report.stop_reason != "completed":
                break
        return report


def _finish_lora(built: Any, config: dict[str, Any], data: Path, out: Path, *, report: Any,
                 fit: Any, dry: bool, export: bool, merge: bool, quant: str,
                 talk: Any) -> dict[str, Any]:
    """Adapter, merge, GGUF, preflight, manifest -- everything after the last step."""
    base = str(built.config.get("base") or "")
    measured = report.seconds / report.steps if report.steps else 0.0
    said = fit
    if measured:
        said = lora.plan(config, base=base,
                             device=str(built.config.get("device") or ""),
                             examples=int(built.config.get("rows") or 0),
                             trainable=int(built.config.get("trainable_parameters") or 0),
                             seconds_per_step=measured, ceiling_min=fit.ceiling_min)
        talk(f"measured: {measured:.1f} s/step over {report.steps} steps"
             + (f" -- {lora.span(measured * fit.steps)} for the {fit.steps} steps this "
                "config asks for" if dry else ""))

    got: dict[str, Any] = {"settings": lora.Lora.of(config).as_dict(),
                           "trainable_parameters": int(
                               built.config.get("trainable_parameters") or 0),
                           "seconds_per_step": round(measured, 3),
                           "plan": said.as_dict()}
    if dry:
        got["adapter"] = ""
        return got

    adapter = lora.save_adapter(built.model, out / ADAPTER_DIR)
    talk(f"adapter: {adapter}")
    got["adapter"] = str(adapter)

    if export or merge:
        merged = lora.merge(base, adapter, out / MERGED_DIR)
        talk(f"merged: {merged}")
        got["merged"] = str(merged)

    if export:
        result = lora.export_gguf(out / MERGED_DIR, out,
                                      name=f"{Path(base).name}-tools", quant=quant)
        talk(f"export: {result.path} ({result.size_mb:.0f} MB)")
        got["gguf"] = str(result.path)
        got["gguf_sha256"] = result.sha256
        try:
            checked = lora.preflight_export(result.path)
            got["preflight"] = lora.summarise(checked)
            got["preflight_ok"] = bool(checked.ok)
        except (RuntimeError, OSError) as exc:
            got["preflight"] = f"not checked: {exc}"
            got["preflight_ok"] = None
        talk(f"preflight: {got['preflight']}")

    manifest = {"recipe": "tool-calls", "base": base, "config": config,
                "data": lora.fingerprint(data), "lora": got,
                "steps": report.steps, "final_loss": report.final_loss,
                "best_metric": report.best_metric, "seconds": round(report.seconds, 1),
                "stop_reason": report.stop_reason}
    write_json(out / "manifest.json", versioned(manifest, MANIFEST_VERSION), default=str)
    talk(f"manifest: {out / 'manifest.json'}")
    return got


def imported(argv: Sequence[str]) -> list[str]:
    """Import every ``--import MODULE`` in ``argv``, so the recipes it registers are known."""
    names = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--import"]
    names += [a.split("=", 1)[1] for a in argv if a.startswith("--import=")]
    for name in names:
        importlib.import_module(name)
    return names


def _config_of(a: Any) -> dict[str, Any]:
    """The settings the flags name, over ``--config``'s file."""
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
    return config


def _parser() -> Any:
    import argparse

    ap = argparse.ArgumentParser(
        prog="ml-stack-train-run", formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Instead of the flags, one of these words:\n"
               "  status   what long runs this machine records -- pid, argv, log\n"
               "  wait     block until the detached training run has ended\n"
               "  stop     end the detached training run\n"
               "  parity   compare this machine's two array backends, operation by "
               "operation\n")
    ap.add_argument("--import", dest="imports", action="append", default=[],
                    metavar="MODULE",
                    help="a module to import so the recipes it registers can be named; "
                         "repeatable")
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
    ap.add_argument("--detach", action="store_true",
                    help=f"run this in the background, owned by nobody's terminal, with "
                         f"its output in a log under {home_dir() / 'logs'}; "
                         f"`ml-stack-train-run status|wait|stop` follow it")
    group = ap.add_argument_group(
        "lora", "train an adapter instead of every weight -- what makes an 8B base "
                "trainable on one machine. Needs peft: pip install 'ml-stack[train-lora]'")
    group.add_argument("--lora", action="store_true", help="train a LoRA adapter")
    group.add_argument("--lora-rank", type=int, default=None)
    group.add_argument("--lora-alpha", type=int, default=None)
    group.add_argument("--lora-dropout", type=float, default=None)
    group.add_argument("--lora-targets", default="", metavar="a,b,c",
                      help="which projections get an adapter; empty means attention and "
                           "MLP both")
    group.add_argument("--merge", action="store_true",
                      help="fold the adapter back into the base, in Hugging Face layout")
    group.add_argument("--export-gguf", action="store_true",
                      help="merge, then convert and quantise into --out, then preflight it")
    group.add_argument("--quant", default="Q8_0", help="the GGUF quantisation to end with")
    group.add_argument("--ceiling", type=float, default=None, metavar="MINUTES",
                      help="refuse a run estimated to take longer than this (default 30)")
    group.add_argument("--yes", action="store_true", help="run past the ceiling")
    return ap


def main(argv: list[str] | None = None) -> int:
    rest = list(sys.argv[1:] if argv is None else argv)
    # The words come before the parser, not through it: --recipe, --data and --out are
    # required to train, and `stop` is asked exactly when there is no recipe to name.
    if rest[:1] == ["wait"]:
        return wait()
    if rest[:1] == ["stop"]:
        return stop()
    if rest[:1] == ["status"]:
        return status()
    if rest[:1] == ["parity"]:
        return parity()

    imported(rest)
    a = _parser().parse_args(rest)

    if a.detach:
        pid = running()
        if pid:
            warn(f"error: a training run (pid {pid}) is still going; "
                 f"`ml-stack-train-run wait` blocks until it has ended, `stop` ends it")
            return 2
        log = detach(rest)
        say(f"detached; the log is {log}")
        say("  ml-stack-train-run status")
        return 0

    stopping = threading.Event()
    previous = signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    try:
        result = run(a.recipe, _config_of(a), Path(a.data), Path(a.out), dry=a.dry_run,
                     export=a.export_gguf, merge=a.merge, quant=a.quant, yes=a.yes,
                     ceiling_min=a.ceiling, should_stop=stopping.is_set)
    except lora.OverCeiling as exc:
        warn(str(exc))
        return 5
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        warn(f"error: {exc}")
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous)
    say(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
