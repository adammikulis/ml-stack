"""Standard benchmark sets -- GSM8K, MMLU-Pro, IFEval, HumanEval -- through
lm-evaluation-harness against any OpenAI-compatible chat endpoint, one JSON per configuration.

``python -m ml_stack.graph.bench.standard --url URL --model NAME [--tasks ...]`` runs each set
as its own `lm_eval.simple_evaluate` call under the bench's measuring lock, times it by wall
clock, and writes::

    {"label", "url", "model", "made_at", "limit", "think",
     "sets": {"gsm8k": {"score", "metric", "filter", "stderr", "n", "seconds", "task"}, ...},
     "harness": {"name": "lm-eval", "version"}}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ml_stack.graph import bench

__all__ = ["SETS", "Set", "HarnessShape", "plan", "summarise", "standard", "main"]

#: The one model type the harness has for a chat endpoint that takes strings, not tokens.
MODEL_TYPE = "local-chat-completions"
#: Seconds the harness waits on one request: a 4k-token answer from a local model is minutes.
TIMEOUT = 3600
#: What is sent to switch thinking on a template that has the switch (Qwen, gemma-4).
THINK_SWITCH = "enable_thinking"
#: Where reasoning ends when a model puts it in the content rather than beside it.
THINK_END = "</think>"
#: The wall clock each set is timed by.
_clock = time.monotonic


class HarnessShape(RuntimeError):
    """The harness's results did not carry the metric a set was expected to report."""


@dataclass(frozen=True)
class Set:
    """One standard set: the harness task that runs it and the ``metric,filter`` it reports."""

    task: str
    metric: str
    filter: str
    max_gen_toks: int = 4096
    unsafe: bool = False

    @property
    def key(self) -> str:
        return f"{self.metric},{self.filter}"


#: Short name -> harness task. Tasks and keys read out of lm-eval 0.4.13's task yamls.
SETS: dict[str, Set] = {
    # 8-shot chain of thought, chat-formatted, answer read after "The final answer is"
    "gsm8k": Set("gsm8k_cot_llama", "exact_match", "strict-match", max_gen_toks=4096),
    # 5-shot chain of thought over 14 subjects; the group row carries the size-weighted mean
    "mmlu_pro": Set("mmlu_pro", "exact_match", "custom-extract", max_gen_toks=4096),
    # 541 prompts, zero-shot; prompt-level strict is the headline number
    "ifeval": Set("ifeval", "prompt_level_strict_acc", "none", max_gen_toks=4096),
    # 164 problems, the instruct variant; runs the model's code through `evaluate`'s
    # code_eval, which needs HF_ALLOW_CODE_EVAL=1 and confirm_run_unsafe_code
    "humaneval": Set("humaneval_instruct", "pass@1", "create_test", max_gen_toks=4096,
                     unsafe=True),
}

DEFAULT_SETS = ("gsm8k", "mmlu_pro", "ifeval", "humaneval")


# -- the harness -------------------------------------------------------------------------

def _evaluate(**kwargs: Any) -> dict[str, Any]:
    """`lm_eval.simple_evaluate(**kwargs)`, imported at the call."""
    from lm_eval import simple_evaluate

    return simple_evaluate(**kwargs)


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("lm-eval")
    except PackageNotFoundError:
        return "not installed"


def plan(*, url: str, model: str, sets: list[str], limit: int | None,
         think: bool | None) -> list[dict[str, Any]]:
    """One `simple_evaluate` kwargs dict per set, in the order given."""
    out = []
    for name in sets:
        one = SETS[name]
        model_args: dict[str, Any] = {
            "base_url": url,
            "model": model,
            "num_concurrent": 1,
            "max_retries": 3,
            "tokenized_requests": False,
            "max_gen_toks": one.max_gen_toks,
            "timeout": TIMEOUT,
        }
        gen_kwargs: dict[str, Any] = {"max_gen_toks": one.max_gen_toks}
        if think is not None:
            gen_kwargs["chat_template_kwargs"] = {THINK_SWITCH: think}
        if think:
            model_args["think_end_token"] = THINK_END
        out.append({
            "set": name,
            "simple_evaluate": {
                "model": MODEL_TYPE,
                "model_args": model_args,
                "tasks": [one.task],
                "limit": limit,
                "apply_chat_template": True,
                "fewshot_as_multiturn": True,
                "gen_kwargs": gen_kwargs,
                "confirm_run_unsafe_code": one.unsafe,
                "log_samples": False,
            },
        })
    return out


def summarise(name: str, results: dict[str, Any], seconds: float) -> dict[str, Any]:
    """One set's row for the JSON, read out of the harness's results dict."""
    one = SETS[name]
    row = results.get("results", {}).get(one.task)
    if row is None or one.key not in row:
        raise HarnessShape(f"{name}: the harness reported no {one.key!r} for {one.task}; "
                           f"it reported {sorted(results.get('results', {}).get(one.task, {}))}")
    stderr = row.get(f"{one.metric}_stderr,{one.filter}")
    return {
        "score": float(row[one.key]),
        "metric": one.metric,
        "filter": one.filter,
        "stderr": None if stderr is None or stderr == "N/A" else float(stderr),
        "n": _scored(one.task, results),
        "seconds": round(seconds, 3),
        "task": one.task,
    }


def _scored(task: str, results: dict[str, Any]) -> int:
    """Documents scored: the task's own count, or the sum over a group's subtasks."""
    counts = results.get("n-samples", {})
    if task in counts:
        return int(counts[task]["effective"])
    children = results.get("group_subtasks", {}).get(task) or []
    return sum(_scored(child, results) for child in children)


def standard(*, url: str, model: str, sets: list[str], limit: int | None, think: bool | None,
             label: str, evaluate: Callable[..., dict[str, Any]] | None = None,
             announce: Callable[[str], None] = print) -> dict[str, Any]:
    """Run every set and return the document; the caller holds the lock and writes it."""
    run = evaluate or _evaluate
    doc: dict[str, Any] = {
        "label": label,
        "url": url,
        "model": model,
        "made_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": limit,
        "think": "server default" if think is None else think,
        "sets": {},
        "harness": {"name": "lm-eval", "version": _version()},
    }
    for step in plan(url=url, model=model, sets=sets, limit=limit, think=think):
        name, kwargs = step["set"], step["simple_evaluate"]
        if SETS[name].unsafe:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        announce(f"{name}: {kwargs['tasks'][0]}"
                 f"{'' if limit is None else f', first {limit}'} ...")
        began = _clock()
        results = run(**kwargs)
        row = summarise(name, results, _clock() - began)
        doc["sets"][name] = row
        announce(f"{name}: {row['score']:.3f} {row['metric']} ({row['filter']}) "
                 f"over {row['n']} in {row['seconds']:.0f} s")
    return doc


# -- the command -------------------------------------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "run"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ml-stack-bench standard", allow_abbrev=False,
        description="Standard benchmark sets through lm-evaluation-harness against an "
                    "OpenAI-compatible chat endpoint; one JSON per configuration.")
    p.add_argument("--url", required=True,
                   help="the chat completions endpoint, "
                        "e.g. http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--model", required=True, help="the model name the endpoint serves")
    p.add_argument("--tasks", default=",".join(DEFAULT_SETS),
                   help="comma-separated, from: gsm8k (gsm8k_cot_llama, 8-shot CoT), "
                        "mmlu_pro (5-shot CoT over 14 subjects), ifeval (541 prompts), "
                        "humaneval (humaneval_instruct, 164 problems; the model's code is "
                        "executed on this machine through `evaluate`'s code_eval in a "
                        "subprocess with HF_ALLOW_CODE_EVAL=1 -- leave it out of --tasks "
                        "to keep generated code from running here). Default: all four")
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N documents of each set (the harness's own "
                        "limit); the JSON records how many were scored")
    p.add_argument("--out", type=Path, default=None,
                   help="where the JSON goes; default <bench home>/standard/<label>-<stamp>.json")
    p.add_argument("--label", default=None, help="a name for this configuration; default the model")
    p.add_argument("--think", action=argparse.BooleanOptionalAction, default=None,
                   help=f"send chat_template_kwargs={{{THINK_SWITCH}: ...}} with every "
                        "request (llama.cpp honours it; a server that does not ignores it); "
                        "neither flag leaves the server's default and says so in the JSON")
    p.add_argument("--dry-run", action="store_true",
                   help="print the simple_evaluate kwargs per set as JSON and write nothing")
    p.add_argument("--no-queue", action="store_true",
                   help="fail with exit 3 if another measurement holds the lock, "
                        "rather than waiting for it")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the sets under the measuring lock and write one JSON; 3 when refused the lock."""
    from ml_stack.lock import Busy, only_one

    args = _parser().parse_args(argv)
    sets = [s.strip() for s in args.tasks.split(",") if s.strip()]
    unknown = [s for s in sets if s not in SETS]
    if unknown:
        print(f"error: unknown set(s) {', '.join(unknown)}; choose from "
              f"{', '.join(SETS)}", file=sys.stderr)
        return 2
    label = args.label or args.model
    steps = plan(url=args.url, model=args.model, sets=sets, limit=args.limit, think=args.think)
    if args.dry_run:
        print(json.dumps(steps, indent=2))
        return 0
    out = args.out or (bench.HOME / "standard" /
                       f"{_slug(label)}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    try:
        with only_one(bench.HOME / "measuring.lock", wait=not args.no_queue,
                      announce=lambda line: print(line, file=sys.stderr)):
            doc = standard(url=args.url, model=args.model, sets=sets, limit=args.limit,
                           think=args.think, label=label)
    except Busy as why:
        print(f"error: {why}. Another measurement is running; wait for it, or pass "
              f"--no-queue to fail fast rather than queue.", file=sys.stderr)
        return 3
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
