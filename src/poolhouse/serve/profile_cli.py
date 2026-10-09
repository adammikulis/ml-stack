"""``poolhouse-serve profile``: the settings a model scored best with for each workload."""

from __future__ import annotations

import argparse
import json

from poolhouse.command import flag, option
from poolhouse.log import say, warn
from poolhouse.serve import ops
from poolhouse.serve.ops import Refused
from poolhouse.serve.profile import ASK, WORKLOADS, said

__all__ = ["OPTIONS", "cmd_profile"]

OPTIONS = [
    flag("model", nargs="?", default="",
         help="a model file, path or hf: reference; every record with none"),
    flag("--for", dest="workload", default="", choices=sorted(WORKLOADS),
         metavar="WORKLOAD",
         help=f"one workload rather than all of them: "
              f"{'; '.join(f'{k}, {v}' for k, v in WORKLOADS.items())}"),
    option("json", help="the records as JSON, exactly as they are kept"),
]


def cmd_profile(args: argparse.Namespace) -> int:
    """``poolhouse-serve profile [MODEL] [--for WORKLOAD]`` -- the settings a model scored
    best in, one block per workload.

    Exit 1 when a model was named and nothing has measured it for any workload.
    """
    model = str(getattr(args, "model", "") or "")
    wanted = str(getattr(args, "workload", "") or "")
    try:
        chosen = ops.servings(model, workload=wanted)
    except Refused as no:
        warn(no.lines[0])
        return 1
    if getattr(args, "json", False):
        say(json.dumps([one.as_dict() for one in chosen], indent=2))
        return 0
    if not chosen:
        say("nothing has been measured yet. `poolhouse-bench sweep` measures one and "
            "`poolhouse-bench report --profile` writes the record.")
        return 0
    say("\n\n".join(said(one) for one in chosen))
    if model and not wanted:
        held = {one.workload for one in chosen}
        for named in (w for w in WORKLOADS if w not in held):
            say(f"\nnothing has measured {model.rsplit('/', 1)[-1]} for {named} "
                f"({WORKLOADS[named]}); `--for {named}` serves the {ASK} record and "
                f"says so")
    return 0
