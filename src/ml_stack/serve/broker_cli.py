"""``ml-stack-serve broker|queue``: run this machine's broker, and see what it holds."""

from __future__ import annotations

import argparse
import json

from ml_stack.command import flag
from ml_stack.log import say
from ml_stack.serve import broker_wire
from ml_stack.serve.broker import IDLE_S, BrokerError
from ml_stack.serve.broker_wire import QUIET_S

__all__ = ["OPTIONS_BROKER", "OPTIONS_QUEUE", "cmd_broker", "cmd_queue"]

OPTIONS_BROKER = [
    flag("--idle", type=float, default=IDLE_S, metavar="SECONDS",
         help="stop a server nobody holds after this long"),
    flag("--quit-after", type=float, default=QUIET_S, metavar="SECONDS",
         help="exit after this long with no server, lease, claim or core grant to look after"),
]

OPTIONS_QUEUE = [
    flag("--json", action="store_true", help="print the broker's snapshot as JSON"),
]


def cmd_broker(args: argparse.Namespace) -> int:
    """``ml-stack-serve broker`` -- run the broker in the foreground."""
    return broker_wire.serve(idle_s=args.idle, quiet_s=args.quit_after)


def cmd_queue(args: argparse.Namespace) -> int:
    """``ml-stack-serve queue`` -- every server the broker holds, who holds it, who waits."""
    try:
        snapshot = broker_wire.status()
    except (BrokerError, OSError) as exc:
        say(str(exc))
        return 1
    if args.json:
        say(json.dumps(snapshot, indent=2))
        return 0
    for held in snapshot["servers"]:
        holders = ", ".join(f"pid {h['pid']}" + (f" ({h['label']})" if h["label"] else "")
                            for h in held["holders"]) or "nobody"
        state = "loading" if held["loading"] else ("ours" if held["ours"] else "not ours")
        say(f":{held['port']}  {held['purpose'] or '-':<10} {held['model']}  [{state}]  "
            f"held by {holders}")
    for at, waiting in enumerate(snapshot["queue"], start=1):
        say(f"waiting #{at}: {waiting['purpose']} {waiting['model']} for pid {waiting['pid']}, "
            f"{waiting['waited_s']}s -- {waiting['blocked_by'] or 'queued'}")
    for name, held in snapshot["claims"].items():
        say(f"claim {name}: pid {held['pid']} {json.dumps(held['info'])}")
    if not (snapshot["servers"] or snapshot["queue"] or snapshot["claims"]):
        say("the broker holds nothing")
    return 0
