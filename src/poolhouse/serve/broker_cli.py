"""``poolhouse-serve broker|queue``: run this machine's broker, and see what it holds."""

from __future__ import annotations

import argparse
import json

from poolhouse.command import flag
from poolhouse.log import say
from poolhouse.serve import broker_wire
from poolhouse.serve.broker import IDLE_S, BrokerError
from poolhouse.serve.broker_wire import QUIET_S

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
    """``poolhouse-serve broker`` -- run the broker in the foreground."""
    return broker_wire.serve(idle_s=args.idle, quiet_s=args.quit_after)


def cmd_queue(args: argparse.Namespace) -> int:
    """``poolhouse-serve queue`` -- every server the broker holds, who holds it, who waits."""
    try:
        snapshot = broker_wire.status()
    except (BrokerError, OSError) as exc:
        say(str(exc))
        return 1
    if args.json:
        say(json.dumps(snapshot, indent=2))
        return 0
    runtime = snapshot.get('runtime') or {}
    environment = runtime.get('environment') or {}
    say(f"broker runtime: {runtime.get('compatibility', 'unknown')}, "
        f"protocol {runtime.get('protocol') or 'unknown'}, "
        f"source {runtime.get('source_commit') or 'unknown'}, "
        f"interpreter {environment.get('interpreter') or 'unknown'}")
    for held in snapshot["servers"]:
        holders = ", ".join(f"pid {h['pid']} ({h['requester']}: {h['reason']})"
                            for h in held["holders"]) or "nobody"
        state = ("loading" if held["loading"] else "unmanaged" if held["unmanaged"]
                 else "ours" if held["ours"] else "adopted, not ours")
        say(f":{held['port']}  {held['purpose'] or '-':<10} {held['model']}  [{held['device'] or '-'}, {state}]  "
            f"held by {holders}")
    for at, waiting in enumerate(snapshot["queue"], start=1):
        say(f"waiting #{at}: {waiting['purpose']} {waiting['model']} for pid {waiting['pid']} "
            f"({waiting['requester']}: {waiting['reason']}), "
            f"{waiting['waited_s']}s -- {waiting['blocked_by'] or 'queued'}")
    for device, line in snapshot.get("requests", {}).items():
        for at, request in enumerate(line):
            say(f"requests {device} #{at}: pid {request.get('pid')} "
                f"({request.get('label') or '-'}) {request.get('url')}"
                + (" -- running" if request.get("running") else " -- waiting"))
    for name, held in snapshot["claims"].items():
        say(f"claim {name}: pid {held['pid']} {json.dumps(held['info'])}")
    if not (snapshot["servers"] or snapshot["queue"] or snapshot["claims"]
            or snapshot.get("requests")):
        say("the broker holds nothing")
    return 0
