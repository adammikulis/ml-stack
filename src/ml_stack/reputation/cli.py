"""``ml-stack-reputation``: look at and delete what ml-stack has recorded about sources.
Person only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from typing import Any

from ml_stack.command import Group, flag, option
from ml_stack.log import say, warn
from ml_stack.memory.vault import KeyUnavailable
from ml_stack.reputation import model
from ml_stack.reputation.sealed import Tampered
from ml_stack.reputation.store import Ledger, Standing
from ml_stack.sentinel.human import HumanRequired, require_person

__all__ = ["COMMAND"]

DENIED = 3
Handler = Callable[[argparse.Namespace, Ledger], int]


def _split(text: str) -> tuple[str, str]:
    kind, _, key = text.partition(":")
    if kind not in model.KINDS or not key:
        raise ValueError(f"name a source as kind:name, kind one of {', '.join(model.KINDS)}")
    return kind, key


def _line(s: Standing) -> str:
    day = time.strftime("%Y-%m-%d", time.gmtime(s.last))
    return (f"{s.kind}:{s.key}  {s.state:<11} short {s.short:.1f}  long {s.long:.1f}  "
            f"clean {s.clean}  {day}{'  NOTICE' if s.notice == 'queued' else ''}")


def _guarded(fn: Handler, *, writes: bool) -> Callable[[argparse.Namespace], int]:
    """``fn`` run only for a person: an agent marker refuses it, and a command that writes
    also needs a terminal on stdin and stdout."""
    def run(args: argparse.Namespace) -> int:
        try:
            require_person(f"ml-stack-reputation {args.cmd}",
                           (sys.stdin.isatty(), True if not writes else sys.stdout.isatty()))
        except HumanRequired as exc:
            warn(str(exc))
            return DENIED
        ledger = Ledger()
        try:
            return fn(args, ledger)
        except (Tampered, KeyUnavailable, ValueError) as exc:
            warn(str(exc))
            return 2
        finally:
            ledger.close()

    return run


def _list(args: argparse.Namespace, ledger: Ledger) -> int:
    rows = ledger.sources()
    if args.json:
        say(json.dumps(ledger.export()["sources"], sort_keys=True))
    else:
        for each in rows:
            say(_line(each))
        if not rows:
            say("(no sources recorded)")
    return 0


def _show(args: argparse.Namespace, ledger: Ledger) -> int:
    kind, key = _split(args.source)
    held = ledger.standing(kind, key)
    if held is None:
        warn(f"no record of {args.source}")
        return 2
    say(json.dumps(next(s for s in ledger.export()["sources"] if s["key"] == held.key
                        and s["kind"] == held.kind), indent=2, sort_keys=True))
    return 0


def _forget(args: argparse.Namespace, ledger: Ledger) -> int:
    if args.all:
        if not args.yes and input("forget every source? [y/N] ").strip().lower() not in ("y", "yes"):
            say("kept")
            return 1
        say(f"forgot {ledger.forget_all()} sources")
        return 0
    if not args.source:
        warn("name a source as kind:name, or --all")
        return 2
    if not ledger.forget(*_split(args.source)):
        warn(f"no record of {args.source}")
        return 2
    say(f"forgot {args.source}")
    return 0


def _export(args: argparse.Namespace, ledger: Ledger) -> int:
    say(json.dumps(ledger.export(), indent=2, sort_keys=True))
    return 0


def _stats(args: argparse.Namespace, ledger: Ledger) -> int:
    stats: dict[str, Any] = ledger.stats()
    say(json.dumps(stats, sort_keys=True) if args.json else
        "\n".join(f"{k}: {v}" for k, v in stats.items()))
    return 0 if stats["status"] != "tampered" else 1


COMMAND = Group("ml-stack-reputation",
                "How every source ml-stack deals with (host, URL, address, fleet peer, model "
                "repository, artifact hash) has behaved: list, show, forget and export what is "
                "recorded, and see its size and integrity. For a person at a terminal; an "
                "agent's process is refused.")
COMMAND.add("list", _guarded(_list, writes=False), help="every source with its state and scores",
            options=[option("json")])
COMMAND.add("show", _guarded(_show, writes=False), help="one source in full",
            options=[flag("source", help="kind:name, for example host:example.org")])
COMMAND.add("forget", _guarded(_forget, writes=True), help="delete one source, or all",
            options=[flag("source", nargs="?", default=""),
                     flag("--all", action="store_true", help="delete the whole store"),
                     option("yes")])
COMMAND.add("export", _guarded(_export, writes=False),
            help="every source as plain JSON on stdout: the only way names leave the encrypted store")
COMMAND.add("stats", _guarded(_stats, writes=False), help="counts by state, limits, integrity",
            options=[option("json")])
main = COMMAND.run
