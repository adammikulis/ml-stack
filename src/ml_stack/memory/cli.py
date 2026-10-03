"""``ml-stack-memory``: look at, add to and delete what the chat agent remembers. Person only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from typing import Any

from ml_stack.command import Group, flag, option
from ml_stack.log import say, warn
from ml_stack.memory.facts import KINDS, SOURCES, Fact, Refused
from ml_stack.memory.store import FACT_LINKS, Store, Tampered
from ml_stack.memory.vault import KeyUnavailable
from ml_stack.sentinel.human import HumanRequired, require_person

__all__ = ["COMMAND"]

DENIED = 3
Handler = Callable[[argparse.Namespace, Store], int]


def _line(store: Store, fact: Fact) -> str:
    why = store.stale(fact)
    day = time.strftime("%Y-%m-%d", time.gmtime(fact.last_confirmed))
    return (f"{fact.id}  {fact.kind:<10} {fact.source:<14} {day}  x{fact.confirm_count}"
            f"{'  RE-CHECK' if why else ''}{'  ' + fact.state.upper() if fact.state != 'current' else ''}"
            f"  {fact.text}")


def _guarded(fn: Handler, *, writes: bool) -> Callable[[argparse.Namespace], int]:
    """``fn`` run only for a person: an agent marker in the environment refuses it, and a
    command that writes also needs a terminal on stdin and stdout."""
    def run(args: argparse.Namespace) -> int:
        try:
            require_person(f"ml-stack-memory {args.cmd}",
                           (sys.stdin.isatty(), True if not writes else sys.stdout.isatty()))
        except HumanRequired as exc:
            warn(str(exc))
            return DENIED
        try:
            store = Store()
            try:
                return fn(args, store)
            finally:
                store.close()
        except (Refused, Tampered, KeyUnavailable, KeyError, ValueError) as exc:
            warn(str(exc.args[0]) if isinstance(exc, KeyError) else str(exc))
            return 2

    return run


def _list(args: argparse.Namespace, store: Store) -> int:
    facts = store.facts()
    if args.json:
        say(json.dumps([f.to_json() for f in facts], sort_keys=True))
        return 0
    for fact in facts:
        say(_line(store, fact))
    if not facts:
        say("(nothing remembered)")
    return 0


def _show(args: argparse.Namespace, store: Store) -> int:
    fact = store.get(args.id)
    if fact is None:
        warn(f"no fact {args.id}")
        return 2
    say(json.dumps({**fact.to_json(), "stale": store.stale(fact)}, indent=2, sort_keys=True))
    return 0


def _add(args: argparse.Namespace, store: Store) -> int:
    models = [f"model:{args.model}"] if args.model else []
    fact = store.add(args.text, args.kind, args.source, entities=[*models, *(args.entity or ())],
                     person=True)
    say(f"remembered {fact.id}")
    return 0


def _forget(args: argparse.Namespace, store: Store) -> int:
    if args.all:
        held = len(store.facts())
        if not args.yes and input(f"forget all {held} facts? [y/N] ").strip().lower() not in ("y", "yes"):
            say("kept")
            return 1
        say(f"forgot {store.forget_all()} facts")
        return 0
    if not args.id:
        warn("name a fact id, or --all")
        return 2
    if not store.forget(args.id):
        warn(f"no fact {args.id}")
        return 2
    say(f"forgot {args.id}")
    return 0


def _confirm(args: argparse.Namespace, store: Store) -> int:
    fact = store.confirm(args.id)
    say(f"{fact.id} confirmed under the build in use now")
    return 0


def _edit(args: argparse.Namespace, store: Store) -> int:
    say(f"edited {store.edit(args.id, args.text).id}")
    return 0


def _link(args: argparse.Namespace, store: Store) -> int:
    store.link(args.id, args.rel, args.other, remove=args.cmd == "unlink")
    say(f"{args.cmd} {args.id} {args.rel} {args.other}")
    return 0


def _rekey(args: argparse.Namespace, store: Store) -> int:
    store.rekey()
    say("re-encrypted under a new key; the old key is gone")
    return 0


def _export(args: argparse.Namespace, store: Store) -> int:
    say(json.dumps(store.export(), indent=2, sort_keys=True))
    return 0


def _stats(args: argparse.Namespace, store: Store) -> int:
    stats: dict[str, Any] = store.stats()
    say(json.dumps(stats, sort_keys=True) if args.json else
        "\n".join(f"{k}: {v}" for k, v in stats.items()))
    return 0 if stats["status"] != "tampered" else 1


COMMAND = Group("ml-stack-memory",
                "What the chat agent remembers across sessions, as an encrypted graph of facts "
                "and the models, builds, settings and tasks they are about: list, show, add, edit, "
                "confirm, link and forget facts, rekey the store, export it, and see its size and "
                "integrity. For a person at a terminal; an agent's process is refused.")
COMMAND.add("list", _guarded(_list, writes=False), help="every fact", options=[option("json")])
COMMAND.add("show", _guarded(_show, writes=False), help="one fact in full",
            options=[flag("id")])
COMMAND.add("add", _guarded(_add, writes=True), help="remember a fact you type",
            options=[flag("text"), flag("--kind", choices=KINDS, default="note"),
                     flag("--source", choices=SOURCES, default="user-said"),
                     flag("--model", default="", help="the model it was learned with"),
                     flag("--entity", action="append", help="what it is about, kind:name "
                          "(model, build, setting, task, topic); repeat for several")])
COMMAND.add("confirm", _guarded(_confirm, writes=True),
            help="mark a fact as checked under the build in use now", options=[flag("id")])
COMMAND.add("forget", _guarded(_forget, writes=True), help="delete one fact, or all",
            options=[flag("id", nargs="?", default=""),
                     flag("--all", action="store_true", help="delete every fact and start a "
                          "new store"), option("yes")])
COMMAND.add("edit", _guarded(_edit, writes=True), help="change the text of a fact",
            options=[flag("id"), flag("text")])
for _name, _help in (("link", "join two facts"), ("unlink", "remove a link between two facts")):
    COMMAND.add(_name, _guarded(_link, writes=True), help=_help,
                options=[flag("id"), flag("rel", choices=FACT_LINKS), flag("other")])
COMMAND.add("rekey", _guarded(_rekey, writes=True),
            help="re-encrypt the store under a new key and drop the old one")
COMMAND.add("export", _guarded(_export, writes=False), help="every fact as plain JSON on stdout: the only way text leaves the encrypted store")
COMMAND.add("stats", _guarded(_stats, writes=False), help="size, limits, staleness, integrity",
            options=[option("json")])
main = COMMAND.run
