"""The ``python -m ml_stack.redteam`` command: run attacks, compare two runs, check the corpus."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from ml_stack.command import Group, flag
from ml_stack.log import say, warn
from ml_stack.redteam import corpus
from ml_stack.redteam.report import (
    DEFAULT_TTD_TOLERANCE,
    Report,
    compare,
    gate,
    markdown,
    success_rates,
)
from ml_stack.redteam.run import DEFAULT_MODEL, Plan, execute
from ml_stack.redteam.scenarios import NAMES, Options

__all__ = ["COMMANDS", "main"]

COMMANDS = Group(
    "python -m ml_stack.redteam",
    "Attack ml-stack's own model-facing surfaces with PyRIT. Traffic stays on this machine "
    "and only installed models are used.", allow_abbrev=False)


TOLERANCES = [
    flag("--success-tolerance", type=float, default=0.0,
         help="the attack-success rate may rise by this much (absolute) before the gate fails; "
              "0 for the deterministic scenarios"),
    flag("--ttd-tolerance", type=int, default=DEFAULT_TTD_TOLERANCE,
         help="sentinel may take this many more tool calls than the baseline to notice an attack "
              "before the gate fails"),
]


def _regressions(old: Report, new: Report, success: float = 0.0,
                 ttd: int = DEFAULT_TTD_TOLERANCE) -> int:
    got = compare(old, new)
    for label in ("regressed", "fixed", "new", "gone"):
        for target, attack_class, attack_id, arm in got[label]:
            say(f"{label:>9}  {target} / {attack_class} / {attack_id}"
                f"{' [' + arm + ']' if arm else ''}")
    before, after, attempts = success_rates(old, new)
    if after > before:
        say(f"success rate rose from {before:.4f} to {after:.4f} over {attempts} attempts")
    worse = gate(old, new, success_tolerance=success, ttd_tolerance=ttd)
    for line in worse:
        say(f"gate: {line}")
    return 1 if worse else 0


@COMMANDS.command("run", help="run scenarios and write a report", options=[
    flag("--scenarios", default=",".join(NAMES), help=f"comma-separated, from {', '.join(NAMES)}"),
    flag("--model", default=DEFAULT_MODEL,
         help="an installed GGUF by name or path, or 'stub' for the scripted stand-in"),
    flag("--stub-mode", choices=("gullible", "resistant"), default="gullible"),
    flag("--limit", type=int, help="most attempts per attack class (smoke runs)"),
    flag("--out", type=Path, default=Path("redteam-report"), help="write OUT.json and OUT.md"),
    flag("--against", type=Path,
         help="a previous report; exit 1 if an attack it blocked now succeeds, the success rate "
              "rose past --success-tolerance, or sentinel got slower than --ttd-tolerance"),
    *TOLERANCES,
])
def _run(args: argparse.Namespace) -> int:
    names = tuple(n for n in args.scenarios.split(",") if n)
    if unknown := sorted(set(names) - set(NAMES)):
        warn(f"unknown scenarios: {', '.join(unknown)}")
        return 2
    plan = Plan(names, args.model, Options(limit=args.limit), args.stub_mode)
    report = execute(plan, shlex.join(["python", "-m", "ml_stack.redteam", *sys.argv[1:]]))
    args.out.with_suffix(".json").write_text(report.to_json(), encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    say(markdown(report))
    return (_regressions(Report.load(args.against), report, args.success_tolerance,
                         args.ttd_tolerance) if args.against else 0)


@COMMANDS.command("compare", help="compare two reports",
                  options=[flag("old", type=Path), flag("new", type=Path), *TOLERANCES])
def _compare(args: argparse.Namespace) -> int:
    return _regressions(Report.load(args.old), Report.load(args.new), args.success_tolerance,
                        args.ttd_tolerance)


@COMMANDS.command("corpus", help="check the attack texts against their recorded hashes")
def _corpus(args: argparse.Namespace) -> int:
    del args
    problems = corpus.verify()
    say("\n".join(problems) or "the corpus matches its manifest")
    return 1 if problems else 0


main = COMMANDS.run
