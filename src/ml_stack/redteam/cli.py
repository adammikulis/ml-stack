"""The ``python -m ml_stack.redteam`` command: run attacks, compare two runs, check the corpus."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from ml_stack.command import Group, flag
from ml_stack.log import say, warn
from ml_stack.redteam import corpus
from ml_stack.redteam.report import Report, compare, markdown
from ml_stack.redteam.run import DEFAULT_MODEL, Plan, execute
from ml_stack.redteam.scenarios import NAMES, Options

__all__ = ["COMMANDS", "main"]

COMMANDS = Group(
    "python -m ml_stack.redteam",
    "Attack ml-stack's own model-facing surfaces with PyRIT. Traffic stays on this machine "
    "and only installed models are used.", allow_abbrev=False)


def _regressions(old: Report, new: Report) -> int:
    got = compare(old, new)
    for label in ("regressed", "fixed", "new", "gone"):
        for target, attack_class, attack_id, arm in got[label]:
            say(f"{label:>9}  {target} / {attack_class} / {attack_id}"
                f"{' [' + arm + ']' if arm else ''}")
    return 1 if got["regressed"] else 0


@COMMANDS.command("run", help="run scenarios and write a report", options=[
    flag("--scenarios", default=",".join(NAMES), help=f"comma-separated, from {', '.join(NAMES)}"),
    flag("--model", default=DEFAULT_MODEL,
         help="an installed GGUF by name or path, or 'stub' for the scripted stand-in"),
    flag("--stub-mode", choices=("gullible", "resistant"), default="gullible"),
    flag("--limit", type=int, help="most attempts per attack class (smoke runs)"),
    flag("--out", type=Path, default=Path("redteam-report"), help="write OUT.json and OUT.md"),
    flag("--against", type=Path,
         help="a previous report; exit 1 if an attack it blocked now succeeds"),
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
    return _regressions(Report.load(args.against), report) if args.against else 0


@COMMANDS.command("compare", help="compare two reports",
                  options=[flag("old", type=Path), flag("new", type=Path)])
def _compare(args: argparse.Namespace) -> int:
    return _regressions(Report.load(args.old), Report.load(args.new))


@COMMANDS.command("corpus", help="check the attack texts against their recorded hashes")
def _corpus(args: argparse.Namespace) -> int:
    del args
    problems = corpus.verify()
    say("\n".join(problems) or "the corpus matches its manifest")
    return 1 if problems else 0


main = COMMANDS.run
