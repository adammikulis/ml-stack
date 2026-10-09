"""``poolhouse-features``: see the experimental features, turn one on or off.

    poolhouse-features list [--json]     every feature with its stage, risk and state
    poolhouse-features enable NAME       turn one on for this machine (audited)
    poolhouse-features disable NAME      turn it off again (audited)

Each takes ``--root DIR`` for a daemon kept somewhere other than the usual place.
"""

from __future__ import annotations

import argparse
import json

from poolhouse import features
from poolhouse.board.client import NodeError
from poolhouse.command import Group, flag, option
from poolhouse.log import say, warn
from poolhouse.testfarm import consent

__all__ = ["GROUP", "main"]

ROOT = flag("--root", default="", help="the daemon directory holding settings.json (default: the usual one)")
NAME = flag("name", help="a feature's name, as `list` shows it")


def _list(args: argparse.Namespace) -> int:
    rows = features.listing(args.root)
    if args.json:
        say(json.dumps(rows, indent=2))
        return 0
    for row in rows:
        say(f"{row['name']}  [{row['stage']}]  {'on' if row['enabled'] else 'off'}")
        say(f"    {row['about']}")
        say(f"    risk: {row['risk']}")
    return 0


def stop_shards() -> None:
    """Turning remote tests off also switches this device's node off, so nothing keeps taking tests behind the feature."""
    try:
        consent.switch("off")
        say("test shards: off")
    except (NodeError, OSError, ValueError) as exc:
        warn(f"the feature is off but this device's node was not switched off ({exc}); run `python -m poolhouse.testfarm.consent off`")


def _switch(on: bool):
    def run(args: argparse.Namespace) -> int:
        try:
            changed = features.switch(args.name, on, root=args.root)
        except features.UnknownFeature as no:
            warn(str(no.args[0]))
            return 2
        say(f"{args.name}: {'on' if on else 'off'}" + ("" if changed else " (already)"))
        if not on and args.name == "remote-tests":
            stop_shards()
        return 0
    return run


GROUP = Group("poolhouse features", "Experimental features: list them, turn one on or off.")
GROUP.add("list", _list, help="every feature with its stage, risk and state", options=[ROOT, option("json")])
GROUP.add("enable", _switch(True), help="turn one on for this machine", options=[ROOT, NAME])
GROUP.add("disable", _switch(False), help="turn one off again", options=[ROOT, NAME])

main = GROUP.run


if __name__ == "__main__":
    raise SystemExit(main())
