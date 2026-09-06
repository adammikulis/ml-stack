"""``ml-stack-world``: invent an organised group, ask it questions, make it talk, export it.

    ml-stack-world make --kind company --size medium --seed 3 --out ./world
    ml-stack-world questions --world ./world --n 40 > questions.jsonl
    ml-stack-world simulate --world ./world --out ./talk --days 20
    ml-stack-world emit --from ./talk --as slack-export --out ./export
    ml-stack-world check ./export ./mail.mbox --truth ./talk

``make`` writes ``graph.json`` (the graph, in the community schema), ``personas.json`` (a
voice, a system prompt and what each person knows), ``calendar.json`` (empty: the simulation
schedules its own arcs into it) and ``world.json`` (kind, size, seed, people), and prints a
summary -- as JSON with ``--json``. ``questions`` writes one question per line in the shape
``ml-stack-bench run`` reads. ``simulate`` is `world.simulate.run`: the people talk for
``--days``, templated unless ``--model-url`` and ``--mix`` hand a share to a model.
``emit`` writes what was said the way a product exports it, so `ml_stack.sources` reads
the invented corpus exactly as it reads a real one. ``check`` is `world.check`: the
corpus read back against the graph the simulation wrote, and every generated name through
the name detector; exit 1 on any miss or hit.

Every subcommand parses and prints; the work is in `ml_stack.world.ops`.
"""

from __future__ import annotations

import argparse
import json
import sys

from ml_stack import home
from ml_stack.command import Group, flag, option
from ml_stack.log import say, warn
from ml_stack.world import ops
from ml_stack.world.check import default_fixtures
from ml_stack.world.ops import EXPORTS, Export, read_messages
from ml_stack.world.organisation import KINDS, SIZES
from ml_stack.world.questions import KINDS as QUESTION_KINDS

__all__ = ["COMMANDS", "EXPORTS", "main", "read_messages"]

COMMANDS = Group(
    "ml-stack-world",
    "Invent an organised group -- a company, a community, a university, an open-source "
    "project or a nonprofit -- as a graph with people who could talk; ask it questions "
    "with known answers; make it talk; export the talk.")
main = COMMANDS.run


@COMMANDS.command(
    "make", help="invent a world and write graph, personas and calendar",
    options=[
        flag("--kind", default="company", choices=KINDS,
             help="what sort of organised group (default: company)"),
        flag("--size", default="small", choices=sorted(SIZES, key=SIZES.get),
             help="how many people: " + ", ".join(f"{s}={n}" for s, n in SIZES.items())),
        flag("--seed", type=int, default=0, help="reproduces the world exactly"),
        option("out", required=True, help="directory to write the files into"),
        option("json", help="print the summary as JSON"),
    ])
def cmd_make(args: argparse.Namespace) -> int:
    made = ops.invent(kind=args.kind, size=args.size, seed=args.seed, out=args.out)
    if args.json:
        say(json.dumps(made, ensure_ascii=False))
        return 0
    say(f"{made['kind']} {made['size']} (seed {made['seed']}): {made['people']} people, "
        f"{made['units']} units, {made['nodes']} nodes, {made['edges']} edges -> "
        f"{made['out']}")
    for relation, count in made["edges_by_relation"].items():
        say(f"  {count:>6}  {relation}")
    return 0


@COMMANDS.command(
    "questions", help="questions with known answers, as bench JSONL",
    options=[
        flag("--world", required=True, help="the directory `make --out` wrote"),
        flag("--n", type=int, default=40, help="how many (default: 40)"),
        flag("--kinds", default="",
             help="only these kinds, comma-separated (any of: "
                  + ", ".join(QUESTION_KINDS) + ")"),
        option("out", help="write here instead of stdout"),
    ])
def cmd_questions(args: argparse.Namespace) -> int:
    try:
        asked = ops.ask(args.world, args.n, args.kinds)
    except ValueError as e:
        warn(str(e))
        return 2
    lines = "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in asked)
    if args.out:
        home.expand(args.out).write_text(lines, encoding="utf-8")
        warn(f"{len(asked)} questions -> {args.out}")
    else:
        sys.stdout.write(lines)
    return 0


@COMMANDS.command(
    "simulate", help="the people talk for some days; writes messages.jsonl",
    options=[
        flag("--world", required=True, help="the directory `make --out` wrote"),
        option("out", required=True,
               help="directory for messages.jsonl, graph.json, calendar.json"),
        flag("--days", type=int, default=20,
             help="working days to simulate (default: 20)"),
        flag("--mix", type=float, default=0.0,
             help="share of threads a model writes, 0-1 (default: 0, all templated)"),
        flag("--model-url", default="", help="a served model, for --mix above 0"),
        flag("--seed", type=int, default=0, help="reproduces the conversations"),
    ])
def cmd_simulate(args: argparse.Namespace) -> int:
    from ml_stack.world.simulate import run

    said = run(args.world, args.out, days=args.days, mix=args.mix,
               model_url=args.model_url or None, seed=args.seed)
    say(json.dumps(said, ensure_ascii=False))
    return 0


@COMMANDS.command(
    "emit", help="write what was said the way a product exports it",
    options=[
        flag("--from", required=True,
             help="the directory `simulate --out` wrote, or a messages.jsonl"),
        flag("--as", required=True, choices=EXPORTS, help="which product's shape"),
        option("out", required=True,
               help="a directory for slack-export; a file for mbox, teams and rows"),
        flag("--world", default="", help="the world directory, for the people's names"),
        flag("--domain", default="example.com",
             help="the domain addresses are minted at"),
        flag("--all", action="store_true",
             help="every message, whichever product it was said in"),
    ])
def cmd_emit(args: argparse.Namespace) -> int:
    where, count = ops.export(Export(
        talk=getattr(args, "from"), shape=getattr(args, "as"), out=args.out,
        world=args.world, domain=args.domain, every=bool(args.all)))
    warn(f"{count} messages -> {where}")
    return 0


@COMMANDS.command(
    "check", help="read an export back against its truth, and run every generated name "
                  "through the name detector",
    options=[
        flag("corpus", nargs="+",
             help="what `emit` wrote: a Slack export directory, an mbox, a Teams JSON or a "
                  "rows JSONL; several to check them together"),
        flag("--truth", required=True,
             help="the directory `simulate --out` wrote (its graph.json holds the "
                  "outcomes), or a graph.json"),
        flag("--fixtures", default=default_fixtures(),
             help="the allow-list of invented names (default: the repository's "
                  "tests/known-fixtures.txt when there is one)"),
        flag("--allow", default=str(home.user_home() / ".config" / "pii-allow.txt"),
             help="a second allow-list (default: ~/.config/pii-allow.txt)"),
        flag("--domain", default="example.com",
             help="the domain the corpus was emitted at"),
    ])
def cmd_check(args: argparse.Namespace) -> int:
    from ml_stack.world import check

    try:
        consistent = check.consistency(args.corpus, args.truth, domain=args.domain)
        private = check.privacy(args.truth, fixtures=args.fixtures, allow=args.allow)
    except (FileNotFoundError, ValueError) as e:
        warn(str(e))
        return 2
    say(check.render(consistent, private))
    return 0 if consistent.ok and private.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
