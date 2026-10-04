"""``ml-stack-log``: read the activity log. Read-only; ``export`` is for a person."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack.activity import query, view, writer
from ml_stack.activity.log import ActivityLog
from ml_stack.activity.schema import Entry
from ml_stack.command import Group, flag
from ml_stack.files import write_text
from ml_stack.keystore import KeystoreError
from ml_stack.log import say, warn
from ml_stack.sentinel.human import AGENT_MARKERS, HumanRequired, require_person

__all__ = ["COMMAND", "main"]

DENIED, BROKEN = 3, 1
FILTERS = [
    flag("--since", default="", metavar="SPAN", help="only records newer than 30m, 2h, 1d"),
    flag("--agent", default="", metavar="NAME", help="records by or addressed to this agent"),
    flag("--kind", default="", metavar="K", help="a kind or a kind prefix, e.g. approval"),
    flag("--session", default="", metavar="S", help="a session id or its first characters"),
    flag("--grep", default="", metavar="TEXT", help="records containing this text"),
    flag("--timeline", action="store_true", help="group each session as a readable story"),
    flag("-n", "--lines", type=int, default=0, help="at most the last N records"),
]


def _filter(args: argparse.Namespace, *, today: bool = False) -> query.Filter:
    since = query.midnight() if today else 0.0
    if args.since:
        since = time.time() - query.duration(args.since)
    return query.Filter(since=since, agent=args.agent, kind=args.kind, session=args.session,
                        grep=args.grep)


def _entries(log: ActivityLog) -> list[Entry]:
    return [e for e in log.entries() if isinstance(e, Entry)]


def _emit(text: str) -> None:
    marked = any(os.environ.get(m) for m in AGENT_MARKERS)
    say(view.fenced(text) if marked else text)


def _shown(args: argparse.Namespace, rows: list[Entry]) -> int:
    rows = rows[-args.lines:] if args.lines else rows
    if not rows:
        say("(no records)")
        return 0
    _emit(view.timeline(rows) if args.timeline else "\n".join(view.line(e) for e in rows))
    return 0


def _guard(fn: Callable[[argparse.Namespace, ActivityLog], int]) -> Callable[[argparse.Namespace], int]:
    def run(args: argparse.Namespace) -> int:
        try:
            return fn(args, writer.log())
        except (KeystoreError, ValueError, OSError) as exc:
            warn(f"{type(exc).__name__}: {exc}")
            return 2
    return run


def _tail(args: argparse.Namespace, log: ActivityLog) -> int:
    flt = _filter(args)
    args.lines = args.lines or 50
    code = _shown(args, query.select(_entries(log), flt))
    if not getattr(args, "follow", False):
        return code
    seen = max((e.seq for e in _entries(log)), default=-1)
    try:
        while True:
            time.sleep(args.interval)
            fresh = [e for e in query.select(_entries(log), flt) if e.seq > seen]
            if fresh:
                seen = fresh[-1].seq
                _emit("\n".join(view.line(e) for e in fresh))
    except KeyboardInterrupt:
        return 0


def _today(args: argparse.Namespace, log: ActivityLog) -> int:
    return _shown(args, query.select(_entries(log), _filter(args, today=True)))


def _show(args: argparse.Namespace, log: ActivityLog) -> int:
    number = args.id.isdigit() and len(args.id) < 8
    hits = [e for e in _entries(log) if (str(e.seq) == args.id if number else e.hash.startswith(args.id))]
    if not hits:
        warn(f"no record {args.id}")
        return 2
    _emit("\n\n".join(view.detail(e) for e in hits[:5]))
    return 0


def _stats(args: argparse.Namespace, log: ActivityLog) -> int:
    rows = list(log.entries())
    held = [e for e in rows if isinstance(e, Entry)]
    sizes = sum(f.stat().st_size for f in log.files())
    extra: dict[str, Any] = {"files": len(log.files()), "bytes": sizes,
                             "unreadable": len(rows) - len(held), "dropped": writer.drops().get("dropped", 0),
                             "head": log.head()}
    _emit(view.stats(held, extra))
    return 0


def _verify(args: argparse.Namespace, log: ActivityLog) -> int:
    done = log.verify(Path(args.anchor).read_text().strip() if args.anchor else "")
    if done.ok:
        say(f"ok: {done.records} records, head {done.head}")
        return 0
    say(f"BROKEN: {done.records} records checked, head {done.head}")
    for problem in done.problems:
        say(f"  {problem}")
    return BROKEN


def _export(args: argparse.Namespace, log: ActivityLog) -> int:
    try:
        require_person("ml-stack-log export", (sys.stdin.isatty(), sys.stdout.isatty()))
    except HumanRequired as exc:
        warn(str(exc))
        return DENIED
    target = Path(args.path).expanduser()
    if target.exists() or log.directory in target.resolve().parents:
        warn(f"{target} exists or is inside the log directory; name a new file elsewhere")
        return 2
    rows = [{"id": e.id, "seq": e.seq, "ts": e.ts, "actor": e.actor, "session": e.session,
             "kind": e.kind, "subject": e.subject, "outcome": e.outcome, "refs": dict(e.refs),
             "meta": dict(e.meta)} for e in query.select(_entries(log), _filter(args))]
    write_text(target, json.dumps(rows, indent=1, sort_keys=True))
    target.chmod(0o600)
    writer.record("activity.export", actor="person", outcome="ok", meta={"records": len(rows)})
    say(f"wrote {len(rows)} records to {target}")
    return 0


def _related(args: argparse.Namespace, log: ActivityLog) -> int:
    rows = query.touching(_entries(log), args.agent_name, args.target)
    return _shown(args, rows)


COMMAND = Group("ml-stack-log",
                "The activity log: what agents, people and the stack did, one encrypted "
                "tamper-evident record each. Read-only; never prints a secret. With no command, "
                "the last 50 records.")
COMMAND.add("tail", _guard(_tail), help="the latest records; -f keeps following",
            options=[*FILTERS, flag("-f", "--follow", action="store_true"),
                     flag("--interval", type=float, default=1.0)])
COMMAND.add("today", _guard(_today), help="records since local midnight", options=FILTERS)
COMMAND.add("show", _guard(_show), help="one record in full by its id or sequence number",
            options=[flag("id")])
COMMAND.add("stats", _guard(_stats), help="counts, span, size, dropped records")
COMMAND.add("verify", _guard(_verify), help="check the chain and say where it breaks",
            options=[flag("--anchor", default="", help="a file holding a head line kept elsewhere")])
COMMAND.add("export", _guard(_export), help="write records as JSON to a new file you name (person only)",
            options=[*FILTERS, flag("--json", action="store_true", required=True),
                     flag("path")])
COMMAND.add("related", _guard(_related), help="everything an agent did that names a subject",
            options=[flag("agent_name", metavar="AGENT"), flag("target", metavar="SUBJECT"),
                     flag("-n", "--lines", type=int, default=0),
                     flag("--timeline", action="store_true")])


def main(argv: list[str] | None = None) -> int:
    """Run the viewer; with no command (or only options) it is ``tail``."""
    items = list(sys.argv[1:] if argv is None else argv)
    if not items or (items[0].startswith("-") and items[0] not in ("-h", "--help")):
        items = ["tail", *items]
    return COMMAND.run(items)
