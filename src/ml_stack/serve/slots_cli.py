"""``ml-stack-serve slots save|restore``: write a running server's slot caches to disk and
read them back, each dump guarded by the layout that wrote it."""

from __future__ import annotations

import argparse
import json

from ml_stack.command import flag, option
from ml_stack.http import ServerError, request_json
from ml_stack.log import say, warn
from ml_stack.serve import ops
from ml_stack.serve.slotdump import SlotGuardRefused, restore_all, save_all

__all__ = ["OPTIONS", "cmd_slots"]

OPTIONS = [
    flag("action", choices=("save", "restore"), help="write the caches out, or read them back"),
    option("port", default=8080, help="the server's port"),
    flag("--slot", action="append", default=[], metavar="NAME=ID",
         help="a named slot to move; every slot the server has when none is given"),
    flag("--dir", dest="directory", default="", metavar="DIR",
         help="the server's --slot-save-path (default: ml-stack's own slots cache)"),
    option("json", help="the dumps as JSON"),
]


def _slots(args: argparse.Namespace) -> dict[str, int]:
    named = {}
    for item in args.slot:
        name, sep, sid = str(item).partition("=")
        if not sep or not sid.strip().isdigit():
            raise ValueError(f"--slot wants NAME=ID, got {item!r}")
        named[name.strip()] = int(sid)
    if named:
        return named
    rows = request_json(f"{ops.base_url_for(args.port)}/slots", method="GET", timeout=10.0)
    return {f"slot{row['id']}": int(row["id"]) for row in rows}


def cmd_slots(args: argparse.Namespace) -> int:
    """``ml-stack-serve slots save|restore`` -- exit 2 when the server or a guard refuses."""
    base = ops.base_url_for(args.port)
    move = save_all if args.action == "save" else restore_all
    try:
        dumps = move(base, _slots(args), directory=args.directory or None)
    except (ServerError, SlotGuardRefused, ValueError) as exc:
        warn(f"error: {exc}")
        return 2
    if args.json:
        say(json.dumps([{"slot": d.slot, "filename": d.filename, "tokens": d.tokens}
                        for d in dumps]))
        return 0
    verb = "saved" if args.action == "save" else "restored"
    for one in dumps:
        say(f"{verb} slot {one.slot}: {one.filename} ({one.tokens} tokens)")
    if not dumps:
        say("no dumps found for these slots")
    return 0
