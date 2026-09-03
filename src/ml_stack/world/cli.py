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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ml_stack.files import write_json
from ml_stack.world import Message
from ml_stack.world.check import default_fixtures
from ml_stack.world.organisation import KINDS, SIZES, load, make, summary
from ml_stack.world.questions import KINDS as QUESTION_KINDS, questions

__all__ = ["EXPORTS", "main", "read_messages"]

EXPORTS = ("slack-export", "mbox", "teams", "rows")
"""What ``emit --as`` can write: a Slack export directory, an mbox, chatMessage JSON as the Graph API returns it,
or a scraper's rows as JSONL."""


def _make(args: argparse.Namespace) -> int:
    world = make(args.kind, args.size, args.seed)
    out = Path(args.out).expanduser()
    write_json(out / "graph.json", world.graph)
    write_json(out / "personas.json", world.personas)
    write_json(out / "calendar.json", world.calendar)
    write_json(out / "world.json", {"kind": world.kind, "size": world.size, "seed": world.seed,
                                    "people": world.people,
                                    "organisation": world.graph["meta"]["world"]["organisation"]})
    made = summary(world)
    made["out"] = str(out)
    if args.json:
        print(json.dumps(made, ensure_ascii=False))
    else:
        print(f"{made['kind']} {made['size']} (seed {made['seed']}): {made['people']} people, "
              f"{made['units']} units, {made['nodes']} nodes, {made['edges']} edges -> {out}")
        for rel, count in made["edges_by_relation"].items():
            print(f"  {count:>6}  {rel}")
    return 0


def _questions(args: argparse.Namespace) -> int:
    world = load(args.world)
    kinds = [k.strip() for k in (args.kinds or "").split(",") if k.strip()]
    try:
        asked = questions(world, args.n, kinds=kinds or None)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    lines = "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in asked)
    if args.out:
        Path(args.out).expanduser().write_text(lines, encoding="utf-8")
        print(f"{len(asked)} questions -> {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(lines)
    return 0


def _simulate(args: argparse.Namespace) -> int:
    from ml_stack.world.simulate import run

    counts = run(args.world, args.out, days=args.days, mix=args.mix,
                 model_url=args.model_url or None, seed=args.seed)
    print(json.dumps(counts, ensure_ascii=False))
    return 0


def read_messages(path: str | Path) -> list[Message]:
    """The ``messages.jsonl`` a simulation wrote, as `Message`s again."""
    out = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["recipients"] = tuple(row.get("recipients") or ())
            out.append(Message(**row))
    return out


def _people_of(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(n["id"]): {"label": str(n.get("label") or "")}
            for n in graph.get("nodes") or () if n.get("kind") == "person"}


def _emit(args: argparse.Namespace) -> int:
    from ml_stack.world import emit

    talk = Path(getattr(args, "from")).expanduser()
    messages = read_messages(talk / "messages.jsonl" if talk.is_dir() else talk)
    # the people are in the simulation's graph.json, or the world's, or the messages
    for where in (talk / "graph.json", Path(args.world).expanduser() / "graph.json" if args.world else None):
        if where and where.exists():
            people = _people_of(json.loads(where.read_text(encoding="utf-8")))
            break
    else:
        people = {m.sender: {} for m in messages}
    out = Path(args.out).expanduser()
    source = None if args.all else {"slack-export": "slack", "mbox": "email", "teams": "teams",
                                    "rows": "slack"}[getattr(args, "as")]
    if getattr(args, "as") == "slack-export":
        where = emit.slack_export(messages, people, out, source=source, domain=args.domain)
    elif getattr(args, "as") == "mbox":
        where = emit.mbox(messages, people, out, source=source, domain=args.domain)
    elif getattr(args, "as") == "teams":
        where = emit.teams(messages, people, out, source=source, domain=args.domain)
    else:
        rows = emit.rows(messages, people, source=source, domain=args.domain)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                       encoding="utf-8")
        where = out
    print(f"{len(messages)} messages -> {where}", file=sys.stderr)
    return 0


def _check(args: argparse.Namespace) -> int:
    from ml_stack.world import check

    try:
        consistent = check.consistency(args.corpus, args.truth, domain=args.domain)
        private = check.privacy(args.truth, fixtures=args.fixtures, allow=args.allow)
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2
    print(check.render(consistent, private))
    return 0 if consistent.ok and private.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ml-stack-world",
        description="Invent an organised group -- a company, a community, a university, an "
                    "open-source project or a nonprofit -- as a graph with people who could "
                    "talk; ask it questions with known answers; make it talk; export the talk.")
    subs = parser.add_subparsers(dest="command", required=True)

    made = subs.add_parser("make", help="invent a world and write graph, personas and calendar")
    made.add_argument("--kind", default="company", choices=KINDS,
                      help="what sort of organised group (default: company)")
    made.add_argument("--size", default="small", choices=sorted(SIZES, key=SIZES.get),
                      help="how many people: " + ", ".join(f"{s}={n}" for s, n in SIZES.items()))
    made.add_argument("--seed", type=int, default=0, help="reproduces the world exactly")
    made.add_argument("--out", required=True, help="directory to write the files into")
    made.add_argument("--json", action="store_true", help="print the summary as JSON")
    made.set_defaults(run=_make)

    asked = subs.add_parser("questions", help="questions with known answers, as bench JSONL")
    asked.add_argument("--world", required=True, help="the directory `make --out` wrote")
    asked.add_argument("--n", type=int, default=40, help="how many (default: 40)")
    asked.add_argument("--kinds", default="",
                       help="only these kinds, comma-separated (any of: " + ", ".join(QUESTION_KINDS) + ")")
    asked.add_argument("--out", default="", help="write here instead of stdout")
    asked.set_defaults(run=_questions)

    talk = subs.add_parser("simulate", help="the people talk for some days; writes messages.jsonl")
    talk.add_argument("--world", required=True, help="the directory `make --out` wrote")
    talk.add_argument("--out", required=True, help="directory for messages.jsonl, graph.json, calendar.json")
    talk.add_argument("--days", type=int, default=20, help="working days to simulate (default: 20)")
    talk.add_argument("--mix", type=float, default=0.0,
                      help="share of threads a model writes, 0-1 (default: 0, all templated)")
    talk.add_argument("--model-url", default="", help="a served model, for --mix above 0")
    talk.add_argument("--seed", type=int, default=0, help="reproduces the conversations")
    talk.set_defaults(run=_simulate)

    export = subs.add_parser("emit", help="write what was said the way a product exports it")
    export.add_argument("--from", required=True,
                        help="the directory `simulate --out` wrote, or a messages.jsonl")
    export.add_argument("--as", required=True, choices=EXPORTS, help="which product's shape")
    export.add_argument("--out", required=True,
                        help="a directory for slack-export; a file for mbox, teams and rows")
    export.add_argument("--world", default="", help="the world directory, for the people's names")
    export.add_argument("--domain", default="example.com", help="the domain addresses are minted at")
    export.add_argument("--all", action="store_true",
                        help="every message, whichever product it was said in")
    export.set_defaults(run=_emit)

    checked = subs.add_parser("check", help="read an export back against its truth, and run "
                                            "every generated name through the name detector")
    checked.add_argument("corpus", nargs="+",
                         help="what `emit` wrote: a Slack export directory, an mbox, a Teams "
                              "JSON or a rows JSONL; several to check them together")
    checked.add_argument("--truth", required=True,
                         help="the directory `simulate --out` wrote (its graph.json holds the "
                              "outcomes), or a graph.json")
    checked.add_argument("--fixtures", default=default_fixtures(),
                         help="the allow-list of invented names (default: the repository's "
                              "tests/known-fixtures.txt when there is one)")
    checked.add_argument("--allow", default=str(Path.home() / ".config" / "pii-allow.txt"),
                         help="a second allow-list (default: ~/.config/pii-allow.txt)")
    checked.add_argument("--domain", default="example.com",
                         help="the domain the corpus was emitted at")
    checked.set_defaults(run=_check)

    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
