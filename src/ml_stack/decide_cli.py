"""``ml-stack-decide``: ask a decision model a question, score one on labelled cases,
calibrate it, and compare backends.

    ml-stack-decide ask "Which team?" --state "payouts failing" --option billing --option sales
    ml-stack-decide ask --state "payouts failing" --noul urgent="Is this urgent?" \\
        --choice team="Which team?:billing,sales,tech" --score anger="How angry?:1..5" --json
    ml-stack-decide eval guards --backend logprob --url http://127.0.0.1:8080
    ml-stack-decide calibrate guards --backend logprob --isotonic --out calibration.json
    ml-stack-decide bench guards --backend logprob --backend embed
"""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import Any

from ml_stack.command import Group, flag, option
from ml_stack.decide import questions, registry, router
from ml_stack.decide.calibrate import Calibration, fit
from ml_stack.decide.cases import Case, fingerprint, read_cases, write_cases
from ml_stack.decide.eval import FLOOR, Report, evaluate, score
from ml_stack.decide.fetch import locate
from ml_stack.decide.guards import guard_cases
from ml_stack.decide.pins import STRANDS_V19
from ml_stack.decide.types import DecideError, Option, options_of
from ml_stack.files import write_json
from ml_stack.log import say, warn
from ml_stack.train import decider_cli as trainer
from ml_stack.train.decider_data import Plan, guard_cases_from_tools
from ml_stack.train.holdout import by_group

__all__ = ["COMMANDS", "main"]

AUTH_ENV = "ML_STACK_DECIDE_KEY"
CASES = [
    flag("cases", help="a JSONL file, or `guards`"), flag("--limit", type=int, default=None),
    flag("--tag", default="", help="only cases carrying this tag (guards: destructive, "
                                   "grounded or scope)"),
    flag("--split", choices=("all", "dev", "test"), default="all",
         help="dev or test: a fixed half of the groups, so tuning never sees the test half"),
]
SERVER = [
    flag("--backend", action="append", default=None,
         help="pointer, logprob, embed or rules; repeat to compare (default: auto)"),
    flag("--url", default="", help=f"chat server for logprob (default ${router.URL_ENV})"),
    flag("--model", default="", help="model name to send, where the server wants one"),
    flag("--embed-url", default="", help="embedding server for the embed backend"),
    flag("--pointer", default="", help="a trained decider: its registered name or directory"),
    flag("--embed-head", default="", help="a trained embed head (.safetensors)"),
    flag("--calibration", default="", help="a calibration file written by `calibrate`"),
    flag("--download", action="store_true",
         help="let the pointer backend download its pinned files"),
]


def _config(args: Namespace) -> router.Config:
    cal = None
    if args.calibration:
        cal = Calibration.from_public(json.loads(Path(args.calibration).read_text())["calibration"])
    return router.Config(backend=(args.backend or ['auto'])[0], url=args.url, model=args.model, token=os.environ.get(AUTH_ENV, ""),
                         embed_url=args.embed_url, embed_head=args.embed_head, pointer=args.pointer,
                         calibration=cal, download=args.download)


def _backends(args: Namespace, count: int, config: router.Config) -> list[str]:
    return list(args.backend) if args.backend else [router.choose(count, config)]


def _text(value: str) -> str:
    if value == "-":
        return sys.stdin.read()
    return Path(value[1:]).read_text() if value.startswith("@") else value


def _option(text: str) -> Option:
    name, _, desc = text.partition("=")
    return Option(name.strip(), desc.strip())


def _cases(args: Namespace) -> list[Case]:
    found = guard_cases() if args.cases == "guards" else read_cases(args.cases)
    if args.split != "all":
        halves = by_group(found, [c.group or c.id for c in found], 0.5, seed=0)
        found = halves.train if args.split == "dev" else halves.holdout
    if args.tag:
        found = [c for c in found if args.tag in c.tags]
    return found[:args.limit] if args.limit else found


def _ask_named(args: Namespace) -> int:
    if args.question or args.option:
        raise ValueError("--noul, --choice and --score cannot be combined with a question or "
                         "--option")
    asked = questions.parse_all(args.noul, args.choice, args.score)
    got = questions.decide(_text(args.state), asked, config=_config(args),
                           abstain_below=args.abstain_below)
    if args.json:
        say(json.dumps({n: a.public() for n, a in got.items()}))
    for name, a in got.items():
        if args.json:
            continue
        if a.error:
            say(f"{name}: ABSTAINED  error: {a.error}")
            continue
        extra = (f"  p_true {a.p_true:.3f}" if a.p_true is not None
                 else f"  expected {a.expected:.3f}" if a.expected is not None else "")
        say(f"{name}: {a.choice}  (certainty {a.certainty:.3f}, {a.backend})" + extra
            + ("  ABSTAINED" if a.abstained else ""))
        for label, p in a.scores.items():
            say(f"  {label:<24} {p:.3f}")
    return 1 if any(a.error for a in got.values()) else 0


def _ask(args: Namespace) -> int:
    if args.noul or args.choice or args.score:
        return _ask_named(args)
    if not args.question or not args.option:
        raise ValueError("ask needs a question and at least two --option, or --noul/--choice/--score")
    config = _config(args)
    opts = options_of([_option(o) for o in args.option])
    name = _backends(args, len(opts), config)[0]
    got = router.build(name, config).decide(args.question, _text(args.state), opts,
                                            abstain_below=args.abstain_below)
    if args.json:
        say(json.dumps(got.public()))
        return 0
    say(f"{got.choice}  (confidence {got.confidence:.3f}, {got.backend}, {got.latency_ms:.0f} ms)"
        + ("  ABSTAINED" if got.abstained else ""))
    for option_name, p in got.scores.items():
        say(f"  {option_name:<24} {p:.3f}")
    return 0


def _reports(args: Namespace, cases: list[Case], config: router.Config) -> list[Report]:
    out = []
    for given in getattr(args, "decider", None) or []:
        pointer = "" if given == "strands" else given
        decider = router.build("pointer", replace(config, backend="pointer", pointer=pointer))
        out.append(evaluate(decider, cases, floor=args.floor))
    if getattr(args, "decider", None):
        return out
    for name in _backends(args, max(len(c.options) for c in cases), config):
        decider = router.build(name, config)
        decider.decide(cases[0].question, cases[0].state, cases[0].options)
        out.append(evaluate(decider, cases, floor=getattr(args, "floor", FLOOR)))
    return out


def _eval(args: Namespace) -> int:
    reports = _reports(args, _cases(args), _config(args))
    for report in reports:
        say(report.line())
    if args.out:
        write_json(Path(args.out), [r.public() for r in reports])
    return 0


def _calibrate(args: Namespace) -> int:
    config = _config(args)
    cases = _cases(args)
    name = _backends(args, max(len(c.options) for c in cases), config)[0]
    split = by_group(cases, [c.group or c.id for c in cases], args.holdout, seed=args.seed)
    decider = router.build(name, config)
    fitting = [decider.decide(c.question, c.state, c.options) for c in split.train]
    rows = [[d.scores[o.name] for o in c.options] for d, c in zip(fitting, split.train, strict=True)]
    cal = fit(rows, [c.label_index for c in split.train], isotonic=args.isotonic)
    held = [decider.decide(c.question, c.state, c.options) for c in split.holdout]
    after = [type(d)(d.choice, dict(zip(d.scores, cal.apply(list(d.scores.values())), strict=True)),
                     d.margin, False, d.latency_ms, d.backend, d.model) for d in held]
    before_report, after_report = score(held, split.holdout), score(after, split.holdout)
    say(f"fitted on {len(split.train)} cases, scored on {len(split.holdout)} held-out "
        f"(whole groups, seed {args.seed})")
    say("before: " + before_report.line())
    say("after:  " + after_report.line())
    write_json(Path(args.out), {"calibration": cal.public(), "backend": name,
                                "model": held[0].model, "data_hash": fingerprint(cases),
                                "holdout": {"before": before_report.public(),
                                            "after": after_report.public()}})
    say(f"wrote {args.out}")
    return 0


def _bench(args: Namespace) -> int:
    reports = _reports(args, _cases(args), _config(args))
    say(f"{'backend':<10} {'n':>4} {'err':>3} {'acc':>6} {'brier':>6} {'ece':>6} "
        f"{'p50 ms':>8} {'p95 ms':>8}")
    for r in reports:
        say(f"{r.backend:<10} {r.n:>4} {r.errors:>3} {r.accuracy:>6.3f} {r.brier:>6.3f} "
            f"{r.ece:>6.3f} {r.latency_ms['p50']:>8.1f} {r.latency_ms['p95']:>8.1f}")
    if args.out:
        write_json(Path(args.out), [r.public() for r in reports])
    return 0


def _export(args: Namespace) -> int:
    say(f"wrote {write_cases(args.out, guard_cases())} cases to {args.out}")
    return 0


def _make(args: Namespace) -> int:
    tools = json.loads(Path(args.tools).read_text())
    tools = tools.get("tools", tools) if isinstance(tools, dict) else tools
    made = guard_cases_from_tools(tools, Plan(seed=args.seed, per_tool=args.per_tool))
    say(f"wrote {write_cases(args.out, made)} synthetic cases for {len(tools)} tools to {args.out}")
    say("their labels come from rules about each tool's name and description: read a sample "
        "before trusting a score measured on them")
    return 0


def _list(args: Namespace) -> int:
    rows = registry.listing()
    for row in rows:
        acc = row["metrics"].get("accuracy")
        say(f"{row['name']:<24} {row['base']:<28} "
            + (f"accuracy {acc:.3f}  " if acc is not None else "") + row["path"])
    if not rows:
        say("no trained deciders; `ml-stack-train-decider` makes one")
    return 0


def _check(args: Namespace) -> int:
    cases = read_cases(args.file)
    labels: dict[str, int] = {}
    for c in cases:
        labels[c.label] = labels.get(c.label, 0) + 1
    say(f"{len(cases)} cases, data hash {fingerprint(cases)[:16]}")
    for label, n in sorted(labels.items()):
        say(f"  {label:<14} {n}")
    return 0


def _fetch(args: Namespace) -> int:
    ck = STRANDS_V19
    say(f"{ck.name}: {len(ck.files)} files, {ck.total_bytes / 1e9:.2f} GB, licence {ck.licence}")
    if not args.yes:
        say("nothing downloaded; add --yes to fetch and verify them")
        return 0
    for pin in ck.files:
        say(f"  {pin.repo}/{pin.filename}  {locate(pin, download=True)}")
    return 0


COMMANDS = Group(
    "ml-stack-decide",
    "Decision models: choose one of a named set of options and say how sure. `ask` poses a "
    "question; `eval` and `bench` score backends on labelled cases (a JSONL file, or "
    "`guards` for the built-in agent-guard set); `calibrate` fits temperature scaling on "
    "held-out groups; `cases` writes and checks case files; `fetch` downloads the pinned "
    "pointer checkpoint.",
    allow_abbrev=False)


@COMMANDS.command("ask", help="pose a question, or named yes/no, choice and score questions", options=[
    flag("question", nargs="?", default="", help="one question, answered among the --option values"),
    flag("--state", default="", help="the situation: text, @file or - (stdin)"),
    flag("--option", action="append", default=[], metavar="NAME[=DESCRIPTION]"),
    flag("--noul", action="append", default=[], metavar="NAME=QUESTION",
         help="a yes/no question; the answer carries p_true"),
    flag("--choice", action="append", default=[], metavar="NAME=QUESTION:OPT1,OPT2",
         help="a question with one answer among the options"),
    flag("--score", action="append", default=[], metavar="NAME=QUESTION:LO..HI",
         help="an integer scale; the answer carries the expected value"),
    flag("--abstain-below", type=float, default=None, help="flag answers under this probability"),
    option("json"), *SERVER])
def ask(args: Namespace) -> int:
    """Pose one question."""
    return _guarded(_ask, args)


@COMMANDS.command("eval", help="score backends on labelled cases", options=[
    *CASES,
    flag("--decider", action="append", default=None, metavar="NAME",
         help="a registered decider, a decider directory or `strands`; repeat to compare"),
    flag("--floor", type=float, default=FLOOR,
         help="confidence under which an answer counts as abstained"),
    option("out"), *SERVER])
def evaluate_cmd(args: Namespace) -> int:
    """Score backends on labelled cases."""
    return _guarded(_eval, args)


@COMMANDS.command("calibrate", help="fit temperature scaling on held-out groups", options=[
    *CASES,
    flag("--isotonic", action="store_true", help="also fit an isotonic map of the top probability"),
    flag("--holdout", type=float, default=0.3, help="share of groups scored, not fitted"),
    flag("--seed", type=int, default=0), option("out", required=True), *SERVER])
def calibrate_cmd(args: Namespace) -> int:
    """Fit a calibration and score it on held-out groups."""
    return _guarded(_calibrate, args)


@COMMANDS.command("bench", help="compare backends: accuracy, Brier, ECE, latency", options=[
    *CASES,
    option("out"), *SERVER])
def bench_cmd(args: Namespace) -> int:
    """Compare backends."""
    return _guarded(_bench, args)


@COMMANDS.command("export-cases", help="write the built-in guard cases as JSONL",
                  options=[option("out", required=True)])
def export_cmd(args: Namespace) -> int:
    """Write the built-in guard cases."""
    return _export(args)


@COMMANDS.command("make-cases", help="synthesise guard cases from an MCP tool list", options=[
    flag("tools", help="a JSON file: the tool list an MCP server returns"),
    flag("--per-tool", type=int, default=8), flag("--seed", type=int, default=0),
    option("out", required=True)])
def make_cmd(args: Namespace) -> int:
    """Synthesise guard cases from a tool list."""
    return _guarded(_make, args)


@COMMANDS.command("train", help="fine-tune, calibrate, register and score a decider",
                  options=trainer.OPTIONS)
def train_cmd(args: Namespace) -> int:
    """Fine-tune a decider from labelled cases."""
    return trainer.run(args)


@COMMANDS.command("list", help="the trained deciders on this machine")
def list_cmd(args: Namespace) -> int:
    """List the registered trained deciders."""
    return _list(args)


@COMMANDS.command("check-cases", help="read a cases file and count its labels",
                  options=[flag("file")])
def check_cmd(args: Namespace) -> int:
    """Count the labels in a cases file."""
    return _check(args)


@COMMANDS.command("fetch", help="download the pinned pointer checkpoint",
                  options=[option("yes")])
def fetch_cmd(args: Namespace) -> int:
    """Download the pinned pointer checkpoint."""
    return _guarded(_fetch, args)


def _guarded(run: Any, args: Namespace) -> int:
    try:
        return run(args)
    except (DecideError, ValueError, OSError) as exc:
        warn(f"error: {exc}")
        return 2


main = COMMANDS.run


if __name__ == "__main__":  # pragma: no cover - the entry point is `ml-stack-decide`
    raise SystemExit(main())
