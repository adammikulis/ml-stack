"""``ml-stack-train-tools``: the three stages, and ``from-bench`` beside them."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ml_stack.log import say, warn
from ml_stack.train.tools.dataset import counts, lines, split, write_dataset
from ml_stack.train.tools.from_bench import from_bench, would_yield
from ml_stack.train.tools.schemas import schemas_of
from ml_stack.train.tools.synthesise import synthesise


def load_tools(spec: str) -> Any:
    """A JSON file of schemas, or ``python:module:attribute`` imported live.

    The attribute may be the schemas, ``(schema, callable)`` pairs, a prompts mapping, or a
    function of no arguments that returns one of those (``ml_stack.web:tools``).
    """
    if spec.startswith("python:"):
        parts = spec.split(":")
        module, attribute = (parts[1], parts[2]) if len(parts) == 3 else ("", "")
        if not module or not attribute:
            raise ValueError(f"expected python:module:attribute, got {spec!r}")
        try:
            value = getattr(importlib.import_module(module), attribute)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"cannot import {spec}: {exc}") from exc
        return value() if callable(value) else value
    path = Path(spec).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"no such file {path}; pass a JSON file or python:module:attr")
    return json.loads(path.read_text())


def _asker(url: str) -> Callable[[str], str]:
    from ml_stack.client import Client

    client = Client(url)

    def ask(prompt: str) -> str:
        reply = client.chat([{"role": "user", "content": prompt}])
        return reply.content or ""

    return ask


def _settings(pairs: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


FROM_BENCH = ("Training data from what a model actually did: every traced question a bench "
              "run kept that scored well enough, as one example per model turn -- the "
              "conversation up to that turn as the input, the call the model made as the "
              "target. Runs are traced by default at 20 questions or fewer "
              "(ml-stack-bench); an untraced run yields nothing, and this says what it "
              "would have yielded.")


def _from_bench_arguments(ap: Any) -> Any:
    """``from-bench``'s flags, onto either parser that has to carry them.

    There are two: the one this subcommand is parsed by, and the one registered under the
    top-level parser so that ``--help`` names it and the documentation check can find its
    flags. Written once so the two cannot drift.
    """
    ap.add_argument("--kept", default="", metavar="STORE",
                    help="the bench store the runs are in (default: ~/.ml-stack/bench/"
                         "runs.ladybug, or MLSTACK_BENCH_HOME's)")
    ap.add_argument("--model", default="", metavar="SUBSTRING",
                    help="only runs whose label or served model file contains this, e.g. "
                         "e4b. Mixing two models' turns into one dataset teaches the "
                         "average of two callers")
    ap.add_argument("--min-f1", type=float, default=0.8, metavar="F1",
                    help="only questions that scored at least this, 0-1 (default: "
                         "%(default)s). A wrong answer's tool calls are exactly what must "
                         "not be learned")
    ap.add_argument("--label", default="", metavar="SUBSTRING",
                    help="only runs whose label contains this (narrower than --model)")
    ap.add_argument("--system", default="", metavar="TEXT",
                    help="a system prompt to put in front of any conversation that has "
                         "none; by default the trace's own is used, which is the one the "
                         "model was actually served")
    ap.add_argument("--base", default="google/functiongemma-270m-it",
                    help="what the manifest names as the model this data is rendered for")
    ap.add_argument("--out", required=True, metavar="FILE.jsonl",
                    help="one JSONL file of rows, or a directory -- which gets train.jsonl, "
                         "holdout.jsonl and manifest.json, ready for ml-stack-train-run "
                         "--recipe tool-calls --data")
    ap.add_argument("--dry-run", action="store_true",
                    help="count what would be written and write nothing")
    return ap


def _from_bench_parser() -> Any:
    return _from_bench_arguments(argparse.ArgumentParser(
        prog="ml-stack-train-tools from-bench", allow_abbrev=False, description=FROM_BENCH))


def _from_bench(argv: list[str]) -> int:
    """``ml-stack-train-tools from-bench``: kept bench traces into a training file."""
    from ml_stack import bench

    a = _from_bench_parser().parse_args(argv)
    store = Path(a.kept).expanduser() if a.kept else bench.home_dir() / "runs.ladybug"
    if not store.exists():
        raise FileNotFoundError(f"no bench store at {store}; run ml-stack-bench first")
    kept = [r for r in bench.runs(store) if not a.label or a.label in str(r.get("label") or "")]
    rows = from_bench(kept, model=a.model, min_f1=a.min_f1, system=a.system)
    could = would_yield(kept, model=a.model, min_f1=a.min_f1)
    say(f"from-bench: {len(kept)} run(s) in {store}, "
        f"{could['questions']} question(s) at or above F1 {a.min_f1:g}"
        + (f" for {a.model!r}" if a.model else "")
        + f", {could['traced']} of them traced")
    if not rows:
        say(f"from-bench: 0 examples. Those questions made {could['turns']} model turns "
            f"between them, which is what a traced run of the same questions would have "
            f"yielded (one example per turn). Trace the next run: a run of "
            f"{bench.SHORT} questions or fewer traces by default, and "
            f"{bench.TRACE_ENV}=1 traces one of any size.")
        return 0
    train, holdout = split(rows)
    say(f"from-bench: {len(rows)} examples from {could['traced']} traced question(s) "
        f"({len(train)} train, {len(holdout)} held out): "
        + ", ".join(f"{k} {v}" for k, v in counts(rows).items()))
    out = Path(a.out).expanduser()
    if a.dry_run:
        say(f"from-bench: would write {out}")
        return 0
    if out.suffix == ".jsonl":
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(lines(rows), encoding="utf-8")
        say(f"from-bench: wrote {out}")
    else:
        summary = write_dataset(out, rows, base=a.base, source=str(store), model=a.model,
                                min_f1=a.min_f1, traced_questions=could["traced"])
        say(f"from-bench: wrote {out}/train.jsonl, holdout.jsonl, manifest.json "
            f"({summary['train']} + {summary['holdout']})")
    return 0


def _parser() -> argparse.ArgumentParser:
    """``ml-stack-train-tools``' own parser, with ``from-bench`` registered under it."""
    ap = argparse.ArgumentParser(
        prog="ml-stack-train-tools", allow_abbrev=False,
        description="Plug in a project's tools, make training data from them, fine-tune a "
                    "base model, end with a GGUF. Each stage is skipped when --out has it. "
                    "`ml-stack-train-tools from-bench --help` builds the same data out of "
                    "what a model actually did, from the traces a bench run kept.")
    ap.add_argument("--tools", required=True,
                    help="JSON list of tool schemas, or python:module:attr "
                         "(e.g. python:ml_stack.graph.prompts:TOOLS)")
    ap.add_argument("--prompts", default="",
                    help="JSON {tool: [question, ...]} or python:module:attr "
                         "(e.g. python:ml_stack.graph.prompts:TOOL_PROMPTS); a 'chat' key is "
                         "the messages that want no tool")
    ap.add_argument("--base", default="google/functiongemma-270m-it",
                    help="Hugging Face id or a local directory to fine-tune")
    ap.add_argument("--out", required=True, help="where data/, run/ and the GGUF go")
    ap.add_argument("--ask", default="", metavar="URL",
                    help="a served model to write more questions per tool")
    ap.add_argument("--per-tool", type=int, default=40, help="conversations per tool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="a tool-calls recipe setting: steps, context, batch_size, "
                         "learning_rate")
    ap.add_argument("--quant", default="Q8_0", help="the GGUF quantisation to end with")
    ap.add_argument("--only", choices=("synth", "train", "export"), default="",
                    help="run one stage")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan with counts; load no model, write nothing")
    # Registered so that --help names it and a reader finds its flags where they look for
    # them. It is not parsed through here: the three stages' --tools and --out are required
    # at this level, and from-bench takes neither. Both carry the same arguments from
    # `_from_bench_arguments`, so the two cannot describe different commands.
    _from_bench_arguments(ap.add_subparsers(dest="cmd", metavar="{from-bench}")
                          .add_parser("from-bench", allow_abbrev=False,
                                      description=FROM_BENCH,
                                      help="training data out of a bench run's traces"))
    return ap


def main(argv: list[str] | None = None) -> int:
    """``ml-stack-train-tools`` -- tool schemas in, a model that calls them out."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "from-bench":
        try:
            return _from_bench(argv[1:])
        except (ValueError, FileNotFoundError, KeyError) as exc:
            warn(f"error: {exc}")
            return 2

    a = _parser().parse_args(argv)
    try:
        return _run(a)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        warn(f"error: {exc}")
        return 2
    except RuntimeError as exc:                       # ToolNotFound, ConversionError
        warn(f"error: {exc}")
        return 2


def _synth(a: Any, data: Path) -> dict[str, Any]:
    """The synth stage: the conversations, written unless they are already there."""
    if (data / "train.jsonl").exists() and not a.dry_run:
        manifest = json.loads((data / "manifest.json").read_text())
        say(f"synth: {manifest['rows']} rows already in {data}, skipping")
        return manifest
    tools = schemas_of(load_tools(a.tools))
    prompts = load_tools(a.prompts) if a.prompts else None
    ask = _asker(a.ask) if a.ask and not a.dry_run else None
    rows = synthesise(tools, prompts=prompts, per_tool=a.per_tool, seed=a.seed, ask=ask)
    train, holdout = split(rows)
    per_tool = counts(rows)
    say(f"synth: {len(rows)} conversations over {len(tools)} tools "
        f"({len(train)} train, {len(holdout)} held out): "
        + ", ".join(f"{k} {v}" for k, v in per_tool.items()))
    if not a.dry_run:
        manifest = write_dataset(data, rows, base=a.base, tools=a.tools,
                                 prompts=a.prompts, seed=a.seed, asked=bool(a.ask))
        say(f"synth: wrote {data}/train.jsonl, holdout.jsonl, manifest.json")
        return manifest
    if a.ask:
        say(f"synth: would also ask {a.ask} for {a.per_tool} more per tool")
    return {"rows": len(rows), "train": len(train), "holdout": len(holdout),
            "per_tool": per_tool}


def _recipe() -> Any:
    """The ``tool-calls`` recipe module, loaded when a stage reaches it."""
    from ml_stack.train.recipes import tool_calls

    return tool_calls


def _train(a: Any, config: dict[str, Any], data: Path, run_dir: Path) -> dict[str, Any]:
    """The train stage: the ``tool-calls`` recipe, resumed from whatever step is there."""
    from ml_stack.train.checkpoint import find_latest, load_state

    latest = find_latest(run_dir)
    done = load_state(latest).step if latest else 0
    plan = {"base": a.base, "steps": config["steps"], "context": config["context"],
            "batch_size": config["batch_size"], "learning_rate": config["learning_rate"],
            "device": str(_recipe().device_for()), "resumed_from": done}
    if done >= config["steps"]:
        say(f"train: {run_dir} already at step {done}, skipping")
        return plan
    if a.dry_run:
        say(f"train: would fine-tune {a.base} for {config['steps']} steps of "
            f"{config['batch_size']} on {plan['device']}, context {config['context']}, "
            f"lr {config['learning_rate']}, into {run_dir}"
            + (f" (resuming from {done})" if done else ""))
        return plan
    from ml_stack.train.run import run

    if not (data / "train.jsonl").exists():
        raise FileNotFoundError(f"no {data}/train.jsonl; run the synth stage first")
    say(f"train: {a.base} for {config['steps']} steps on {plan['device']}"
        + (f", resuming from {done}" if done else ""))
    plan.update(run("tool-calls", config, data, run_dir))
    say(f"train: final loss {plan['final_loss']:.3f}, best held-out "
        f"{plan['best_metric']}, {plan['seconds']}s")
    return plan


def _export(a: Any, out: Path, run_dir: Path) -> dict[str, Any]:
    """The export stage: the checkpoint back into Hugging Face layout, then a GGUF."""
    from ml_stack.gguf.tools import find_converter, find_quantize

    existing = sorted(out.glob("*.gguf"))
    converter, quantizer = find_converter(), find_quantize()
    plan = {"quant": a.quant, "converter": str(converter or ""),
            "quantize": str(quantizer or "")}
    if existing:
        say(f"export: {existing[0]} already exists, skipping")
        plan["gguf"] = str(existing[0])
        return plan
    if a.dry_run:
        say(f"export: would write {out / 'model'} in Hugging Face layout, then "
            f"{a.quant} GGUF via " + (str(converter) if converter else
                                        "convert_hf_to_gguf.py, which is NOT installed"))
        return plan
    from ml_stack.gguf import export

    saved = _recipe().save_pretrained(run_dir, a.base, out / "model")
    say(f"export: wrote {saved}")
    result = export(saved, out, name=f"{Path(a.base).name}-tools", quant=a.quant,
                    fix_space_prefix=None)
    say(f"export: {result.path} ({result.size_mb:.0f} MB)")
    plan["gguf"] = str(result.path)
    return plan


def _run(a: Any) -> int:
    """The three stages in order, each skipped when ``--out`` already holds its output."""
    from ml_stack.train.recipes import validate

    out = Path(a.out).expanduser()
    data, run_dir = out / "data", out / "run"
    stages = (a.only,) if a.only else ("synth", "train", "export")
    config = validate("tool-calls", _settings(a.set))
    summary: dict[str, Any] = {"out": str(out), "base": a.base, "dry_run": a.dry_run}

    if "synth" in stages:
        summary["synth"] = _synth(a, data)
    if "train" in stages:
        summary["train"] = _train(a, config, data, run_dir)
    if "export" in stages:
        summary["export"] = _export(a, out, run_dir)

    say(json.dumps(summary, indent=2))
    return 0
