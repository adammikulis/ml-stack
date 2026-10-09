"""``ml-stack-features``: see the experimental features, turn one on or off.

    ml-stack-features list [--json]     every feature with its stage, risk and state
    ml-stack-features enable NAME       turn one on for this machine (audited)
    ml-stack-features disable NAME      turn it off again (audited)

Each takes ``--root DIR`` for a daemon kept somewhere other than the usual place.
"""

from __future__ import annotations

import argparse
import json

from ml_stack import features
from ml_stack.command import Group, flag, option
from ml_stack.log import say, warn

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


def _switch(on: bool):
    def run(args: argparse.Namespace) -> int:
        try:
            changed = features.switch(args.name, on, root=args.root)
        except features.UnknownFeature as no:
            warn(str(no.args[0]))
            return 2
        say(f"{args.name}: {'on' if on else 'off'}" + ("" if changed else " (already)"))
        return 0
    return run


GROUP = Group("ml-stack features", "Experimental features: list them, turn one on or off.")
GROUP.add("list", _list, help="every feature with its stage, risk and state", options=[ROOT, option("json")])
GROUP.add("enable", _switch(True), help="turn one on for this machine", options=[ROOT, NAME])
GROUP.add("disable", _switch(False), help="turn one off again", options=[ROOT, NAME])

main = GROUP.run


if __name__ == "__main__":
    raise SystemExit(main())
