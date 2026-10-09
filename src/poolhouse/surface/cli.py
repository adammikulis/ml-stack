"""``poolhouse-surface`` -- every command's help and output, captured and compared.

    poolhouse-surface capture --out ./before
    poolhouse-surface capture --out ./after
    poolhouse-surface diff ./before ./after

``capture`` walks every ``poolhouse-<command>`` and every subcommand each declares, writes
``--help`` for all of them and the output of the ones that are safe to run, and prints
what it did. ``list`` prints the same walk without running anything. ``diff`` prints what
changed between two captures and exits 1 when anything did.

Every subcommand parses and prints; the work is in `poolhouse.surface.ops`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from poolhouse import home
from poolhouse.command import Group, flag, option
from poolhouse.log import say, warn
from poolhouse.surface import ops
from poolhouse.surface.ops import SurfaceFailed

__all__ = ["COMMANDS", "main"]

COMMANDS = Group(
    "poolhouse-surface",
    "Every command's --help and every safe command's output, written to a directory and "
    "compared with another -- so a refactor is diffed rather than eyeballed.")
main = COMMANDS.run


def _src(named: str) -> Path:
    return home.expand(named) if named else ops.SRC


def _only(named: str) -> list[str]:
    return [word.strip() for word in named.split(",") if word.strip()]


@COMMANDS.command(
    "capture", help="write every command's help and every safe invocation's output",
    options=[
        option("out", required=True, help="directory to write the capture into"),
        flag("--src", default="", metavar="DIR",
             help="the src/ of the checkout to ask (default: the one this is running from)"),
        flag("--only", default="", metavar="WORDS",
             help="only these commands, comma-separated -- `serve,bench`; `` is poolhouse "
                  "itself"),
        flag("--help-only", action="store_true", dest="help_only",
             help="capture no invocations at all, only --help"),
        option("timeout", default=120.0, help="seconds any one command may take"),
    ])
def cmd_capture(args: argparse.Namespace) -> int:
    out = home.expand(args.out)
    src = _src(args.src)
    try:
        taken = ops.capture(out, src=src, only=_only(args.only), help_only=args.help_only,
                            timeout=args.timeout)
    except SurfaceFailed as refused:
        warn(f"error: {refused}")
        return 2
    counts: dict[str, int] = {}
    for one in taken.runs:
        counts[one.verdict] = counts.get(one.verdict, 0) + 1
    helped = [node.spoken or "poolhouse" for node in taken.help_only]
    say(f"{len(taken.nodes)} commands and subcommands from {src}, written to {out}")
    say(f"  {len(taken.nodes)} --help")
    for verdict in sorted(counts):
        say(f"  {counts[verdict]:>3} {verdict}")
    say(f"  help only: {', '.join(helped) or 'none'}")
    return 0


@COMMANDS.command(
    "list", help="every command and subcommand, and what a capture would take of each",
    options=[
        flag("--src", default="", metavar="DIR", help="the src/ of the checkout to ask"),
        flag("--only", default="", metavar="WORDS", help="only these commands"),
        option("timeout", default=120.0, help="seconds any one command may take"),
    ])
def cmd_list(args: argparse.Namespace) -> int:
    try:
        nodes = ops.walk(_src(args.src), _only(args.only), args.timeout)
    except SurfaceFailed as refused:
        warn(f"error: {refused}")
        return 2
    runs = {one.slug: one for one in ops.invocations(nodes)}
    lines = [(node.spoken or "poolhouse", "--help, and " + _verdict(runs.get(node.slug)))
             for node in nodes]
    lines += [(" ".join(one.words), _verdict(one))
              for slug, one in sorted(runs.items())
              if slug not in {node.slug for node in nodes}]
    width = max((len(said) for said, _ in lines), default=0)
    for said, verdict in lines:
        say(f"{said:<{width}}  {verdict}")
    return 0


def _verdict(run: ops.Run | None) -> str:
    """What a capture takes of one node, in the words `list` prints."""
    if run is None:
        return "nothing run -- not declared safe"
    if run.verdict == "refused":
        return "runs it bare for the usage it is refused with"
    return "runs " + " ".join(["poolhouse", *run.words])


@COMMANDS.command(
    "diff", help="what changed between two captures; exit 1 when anything did",
    options=[
        flag("before", metavar="BEFORE", help="a directory `capture` wrote"),
        flag("after", metavar="AFTER", help="another one to compare it with"),
        flag("--lines", type=int, default=2, metavar="N",
             help="lines of context around each change (default: 2)"),
    ])
def cmd_diff(args: argparse.Namespace) -> int:
    before, after = home.expand(args.before), home.expand(args.after)
    changed = ops.compare(before, after, args.lines)
    if not changed:
        say(f"nothing differs between {before} and {after}")
        return 0
    for one in changed:
        say(f"\n{one.name}")
        say(one.diff)
    say(f"\n{len(changed)} captures differ")
    return 1


if __name__ == "__main__":  # pragma: no cover - the entry point is `poolhouse-surface`
    raise SystemExit(main())
