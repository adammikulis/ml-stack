"""``ml-stack-memory``: look at, add to and delete what the chat agent remembers. Person only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import authority
from ml_stack.command import Group, flag, option
from ml_stack.log import say, warn
from ml_stack.memory.facts import KINDS, SOURCES, Fact, Refused
from ml_stack.memory.project import SCOPES, at
from ml_stack.memory.store import FACT_LINKS, Setup, Store, Tampered
from ml_stack.memory.union import Memory
from ml_stack.memory.vault import KeyUnavailable
from ml_stack.person import HumanRequired, is_terminal

__all__ = ["COMMAND"]

DENIED = 3
Handler = Callable[[argparse.Namespace, Memory], int]


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
            authority.require("memory.admin", f"ml-stack-memory {args.cmd}",
                              (is_terminal(sys.stdin), True if not writes else is_terminal(sys.stdout)))
        except HumanRequired as exc:
            warn(str(exc))
            return DENIED
        try:
            mem = Memory.open(explicit=Path(args.project) if args.project else None)
            try:
                return fn(args, mem)
            finally:
                mem.close()
        except (Refused, Tampered, KeyUnavailable, KeyError, ValueError) as exc:
            warn(str(exc.args[0]) if isinstance(exc, KeyError) else str(exc))
            return 2

    return run


def _chosen(args: argparse.Namespace, mem: Memory) -> list[Store]:
    """The stores ``--scope`` names, or both when it is not given."""
    return [mem.store(args.scope)] if args.scope else list(mem.stores.values())


def _owner(args: argparse.Namespace, mem: Memory, ident: str) -> Store:
    """The one store holding fact ``ident``; ``--scope`` decides when both do."""
    held = [s for s in _chosen(args, mem) if s.get(ident) is not None]
    if len(held) != 1:
        raise KeyError(f"no fact {ident}" if not held else
                       f"{ident} is in both scopes: add --scope user or --scope project")
    return held[0]


def _list(args: argparse.Namespace, mem: Memory) -> int:
    pairs = [(s, f) for s in _chosen(args, mem) for f in s.facts()]
    if args.json:
        say(json.dumps([{**f.to_json(), "scope": s.realm} for s, f in pairs], sort_keys=True))
        return 0
    for store, fact in pairs:
        say(f"{store.realm:<8}{_line(store, fact)}")
    if not pairs:
        say("(nothing remembered)")
    return 0


def _show(args: argparse.Namespace, mem: Memory) -> int:
    store = _owner(args, mem, args.id)
    fact = store.get(args.id)
    say(json.dumps({**(fact.to_json() if fact else {}), "scope": store.realm, "stale": store.stale(fact) if fact else ""},
                   indent=2, sort_keys=True))
    return 0


def _add(args: argparse.Namespace, mem: Memory) -> int:
    models = [f"model:{args.model}"] if args.model else []
    fact = mem.store(args.scope).add(args.text, args.kind, args.source,
                                     entities=[*models, *(args.entity or ())], person=True)
    say(f"remembered {fact.id} ({args.scope})")
    return 0


def _forget(args: argparse.Namespace, mem: Memory) -> int:
    if args.all:
        scopes = [args.scope] if args.scope else (
            [input(f"forget all facts in which scope? [{'/'.join(mem.stores)}/both] ").strip().lower()]
            if not args.yes else [])
        if scopes == ["both"]:
            scopes = list(mem.stores)
        if not scopes or any(x not in mem.stores for x in scopes):
            warn("name the scope: --scope user or --scope project")
            return 2
        held = sum(len(mem.store(x).facts()) for x in scopes)
        if not args.yes and input(f"forget all {held} facts in {' and '.join(scopes)}? [y/N] ").strip().lower() not in ("y", "yes"):
            say("kept")
            return 1
        say(f"forgot {sum(mem.store(x).forget_all() for x in scopes)} facts")
        return 0
    if not args.id:
        warn("name a fact id, or --all")
        return 2
    if not _owner(args, mem, args.id).forget(args.id):
        warn(f"no fact {args.id}")
        return 2
    say(f"forgot {args.id}")
    return 0


def _confirm(args: argparse.Namespace, mem: Memory) -> int:
    fact = _owner(args, mem, args.id).confirm(args.id)
    say(f"{fact.id} confirmed under the build in use now")
    return 0


def _edit(args: argparse.Namespace, mem: Memory) -> int:
    say(f"edited {_owner(args, mem, args.id).edit(args.id, args.text).id}")
    return 0


def _link(args: argparse.Namespace, mem: Memory) -> int:
    _owner(args, mem, args.id).link(args.id, args.rel, args.other, remove=args.cmd == "unlink")
    say(f"{args.cmd} {args.id} {args.rel} {args.other}")
    return 0


def _rekey(args: argparse.Namespace, mem: Memory) -> int:
    mem.rekey()
    say("re-encrypted under a new key; the old key is gone")
    return 0


def _export(args: argparse.Namespace, mem: Memory) -> int:
    out = {s.realm: s.export() for s in _chosen(args, mem)}
    say(json.dumps(out[args.scope] if args.scope else out, indent=2, sort_keys=True))
    return 0


def _stats(args: argparse.Namespace, mem: Memory) -> int:
    stats: list[dict[str, Any]] = [s.stats() for s in _chosen(args, mem)]
    say(json.dumps(stats, sort_keys=True) if args.json else
        "\n\n".join("\n".join(f"{k}: {v}" for k, v in one.items()) for one in stats))
    return 0 if all(one["status"] != "tampered" for one in stats) else 1


def _projects(args: argparse.Namespace, mem: Memory) -> int:
    rows = [(s.path.parent.name, s.saved_project.get("name", "?"), s.saved_project.get("root", "?"),
             len(s.facts())) for s in mem.on_disk()]
    for key, name, root, held in rows:
        say(f"{key}  {name}  {root}  {held} facts")
    if not rows:
        say("(no project memory)")
    return 0


def _relink(args: argparse.Namespace, mem: Memory) -> int:
    if mem.project is None:
        warn("no project is open here; run it inside the project or give --project PATH")
        return 2
    old = Store(setup=Setup(user=mem.user.user, profile=mem.user.profile, keys=mem.user.keys,
                            project=at(Path(args.old))))
    try:
        moved = mem.project.adopt(old)
        old.forget_all()
    finally:
        old.close()
    say(f"moved {moved} facts from {args.old} to {mem.project.project.name if mem.project.project else ''}")
    return 0


COMMAND = Group("ml-stack-memory",
                "What the chat agent remembers across sessions, as an encrypted graph of facts "
                "and the models, builds, settings and tasks they are about: list, show, add, edit, "
                "confirm, link and forget facts, rekey the store, export it, and see its size and "
                "integrity, in two scopes: yours (all projects) and the project's (--project PATH, "
                "else the git repository or directory you are in). For a person at a terminal; an "
                "agent's process is refused.")
SCOPE = flag("--scope", choices=SCOPES, default="", help="user or project (default: both)")
PROJECT = flag("--project", default="", metavar="PATH", help="the project whose memory is used (default: "
               "the git repository or directory you are in)")
COMMON = [SCOPE, PROJECT]
ADD_SCOPE = flag("--scope", choices=SCOPES, required=True, help="user (about you, all projects) or project")
COMMAND.add("list", _guarded(_list, writes=False), help="every fact", options=[option("json"), *COMMON])
COMMAND.add("show", _guarded(_show, writes=False), help="one fact in full", options=[flag("id"), *COMMON])
COMMAND.add("add", _guarded(_add, writes=True), help="remember a fact you type",
            options=[flag("text"), ADD_SCOPE, PROJECT, flag("--kind", choices=KINDS, default="note"),
                     flag("--source", choices=SOURCES, default="user-said"),
                     flag("--model", default="", help="the model it was learned with"),
                     flag("--entity", action="append", help="what it is about, kind:name "
                          "(model, build, setting, task, topic); repeat for several")])
COMMAND.add("confirm", _guarded(_confirm, writes=True),
            help="mark a fact as checked under the build in use now", options=[flag("id"), *COMMON])
COMMAND.add("forget", _guarded(_forget, writes=True), help="delete one fact, or all in a scope",
            options=[flag("id", nargs="?", default=""),
                     flag("--all", action="store_true", help="delete every fact in a scope and "
                          "start a new store (asks which scope unless --scope is given)"),
                     option("yes"), *COMMON])
COMMAND.add("edit", _guarded(_edit, writes=True), help="change the text of a fact",
            options=[flag("id"), flag("text"), *COMMON])
for _name, _help in (("link", "join two facts"), ("unlink", "remove a link between two facts")):
    COMMAND.add(_name, _guarded(_link, writes=True), help=_help,
                options=[flag("id"), flag("rel", choices=FACT_LINKS), flag("other"), *COMMON])
COMMAND.add("rekey", _guarded(_rekey, writes=True),
            help="re-encrypt the user store and every project store under one new key and drop the old one",
            options=[PROJECT])
COMMAND.add("export", _guarded(_export, writes=False), options=COMMON,
            help="facts as plain JSON on stdout: the only way text leaves the encrypted stores")
COMMAND.add("stats", _guarded(_stats, writes=False), help="size, limits, staleness, integrity of each scope",
            options=[option("json"), *COMMON])
COMMAND.add("projects", _guarded(_projects, writes=False), help="the projects that have memory", options=[PROJECT])
COMMAND.add("relink", _guarded(_relink, writes=True), options=[flag("old", metavar="OLD_PATH"), PROJECT],
            help="move the memory of a project that was at OLD_PATH to the project you are in")
main = COMMAND.run
