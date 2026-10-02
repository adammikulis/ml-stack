"""The ``python -m ml_stack.redteam`` command: run attacks, compare two runs, check the corpus."""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from ml_stack.redteam import corpus
from ml_stack.redteam.report import Report, compare, markdown
from ml_stack.redteam.run import DEFAULT_MODEL, Plan, execute
from ml_stack.redteam.scenarios import NAMES, Options

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml_stack.redteam",
        description="Attack ml-stack's own model-facing surfaces with PyRIT. Traffic stays on "
                    "this machine and only installed models are used.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run scenarios and write a report")
    run.add_argument("--scenarios", default=",".join(NAMES),
                     help=f"comma-separated, from {', '.join(NAMES)}")
    run.add_argument("--model", default=DEFAULT_MODEL,
                     help="an installed GGUF by name or path, or 'stub' for the scripted stand-in")
    run.add_argument("--stub-mode", choices=("gullible", "resistant"), default="gullible")
    run.add_argument("--limit", type=int, help="most attempts per attack class (smoke runs)")
    run.add_argument("--out", type=Path, default=Path("redteam-report"),
                     help="write OUT.json and OUT.md")
    run.add_argument("--against", type=Path,
                     help="a previous report; exit 1 if an attack it blocked now succeeds")
    diff = sub.add_parser("compare", help="compare two reports")
    diff.add_argument("old", type=Path)
    diff.add_argument("new", type=Path)
    sub.add_parser("corpus", help="check the attack texts against their recorded hashes")
    return parser


def _regressions(old: Report, new: Report) -> int:
    got = compare(old, new)
    for label in ("regressed", "fixed", "new", "gone"):
        for target, attack_class, attack_id, arm in got[label]:
            print(f"{label:>9}  {target} / {attack_class} / {attack_id}"
                  f"{' [' + arm + ']' if arm else ''}")
    return 1 if got["regressed"] else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "corpus":
        problems = corpus.verify()
        print("\n".join(problems) or "the corpus matches its manifest")
        return 1 if problems else 0
    if args.command == "compare":
        return _regressions(Report.load(args.old), Report.load(args.new))
    names = tuple(n for n in args.scenarios.split(",") if n)
    if unknown := sorted(set(names) - set(NAMES)):
        print(f"unknown scenarios: {', '.join(unknown)}", file=sys.stderr)
        return 2
    plan = Plan(names, args.model, Options(limit=args.limit), args.stub_mode)
    report = execute(plan, shlex.join(["python", "-m", "ml_stack.redteam", *(argv or sys.argv[1:])]))
    args.out.with_suffix(".json").write_text(report.to_json(), encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return _regressions(Report.load(args.against), report) if args.against else 0
