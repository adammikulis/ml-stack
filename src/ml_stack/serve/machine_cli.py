"""``ml-stack-serve memory|limits|reclaim``: how much of this machine ml-stack may take,
and stopping the servers nobody is using."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ml_stack.command import flag
from ml_stack.log import say, warn
from ml_stack.serve import ops
from ml_stack.serve.ops import Refused
from ml_stack.units import human_bytes

__all__ = ["OPTIONS_LIMITS", "OPTIONS_MEMORY", "OPTIONS_RECLAIM", "cmd_limits",
           "cmd_memory", "cmd_reclaim"]

OPTIONS_MEMORY = [
    flag("--persist", nargs="?", const="", default=None, metavar="MB",
         help="write a boot-time setting for the wiring limit; the megabytes default "
              "to whatever is set now"),
    flag("--write", default="", metavar="FILE",
         help="where to write it (default: ./stack.ml.wired-limit.plist)"),
    flag("--limit", type=int, default=0, metavar="MB",
         help="preview: what the rest of the machine would have under this wiring "
              "limit, against what it holds now"),
]


def cmd_memory(args: argparse.Namespace) -> int:
    """``ml-stack-serve memory`` -- what this machine will let a model use, and for how long.

    On unified memory the ceiling that matters is `iogpu.wired_limit_mb`: a runtime setting
    that goes back to the default on every reboot.
    """
    machine = ops.memory()
    total, now = machine.total, machine.room
    if not now:
        say("this machine does not report a wiring limit; nothing to do here")
        return 0

    say(f"a model may use about {human_bytes(now)}"
        + (f" of {human_bytes(total)} installed" if total else ""))
    if total:
        default = int(total * 0.75)
        if now > default * 1.02:
            say(f"  raised from the ~{human_bytes(default)} default -- and **not** kept: this "
                f"resets on reboot")
        else:
            say(f"  this is the default share; {human_bytes(total)} is installed")

    held = machine.held
    if held:
        say(f"\nright now: {human_bytes(held['used'])} used of {human_bytes(held['total'])} "
            f"({human_bytes(held['wired'])} wired, {human_bytes(held['free'])} free)")
        servers = held["servers"]
        others = held["others"]
        say(f"  llama-server(s): {human_bytes(servers)}; everything else: {human_bytes(others)}"
            + (f" -- {', '.join(held['largest'])}" if held["largest"] else ""))
        headroom = int(total) - int(now) if total else 0
        if total:
            say(f"  the limit leaves {human_bytes(headroom)} for everything else; the rest of "
                f"the machine holds {human_bytes(others)} now"
                + (" -- room to raise it" if others < headroom * 0.6
                   else " -- close to it; raising it means swapping when a model fills it"))
        want_mb = int(getattr(args, "limit", 0) or 0)
        if want_mb and total:
            left = int(total) - want_mb * 1024 * 1024
            say(f"  at {want_mb} MB the rest of the machine would have {human_bytes(left)}"
                + (" -- less than it holds now" if left < others else ""))
    want = args.persist
    if want is None:
        say("\n  ml-stack-serve memory --persist [MB]   to write a boot-time setting")
        return 0

    mb = int(want) if want else now // (1024 * 1024)
    where = ops.write_plist(Path(args.write or "./stack.ml.wired-limit.plist"), mb)
    say(f"\nwrote {where} -- it sets iogpu.wired_limit_mb={mb} at every boot.")
    say("Installing it needs root, so it is left to you:")
    say(f"  sudo cp {where} /Library/LaunchDaemons/stack.ml.wired-limit.plist")
    say("  sudo chown root:wheel /Library/LaunchDaemons/stack.ml.wired-limit.plist")
    say("  sudo launchctl load -w /Library/LaunchDaemons/stack.ml.wired-limit.plist")
    say(f"\nOr for this boot only:  sudo sysctl -w iogpu.wired_limit_mb={mb}")
    say("\nLeave headroom: everything else on the machine shares this memory, and a "
        "machine that wires all of it stops being usable before it stops serving.")
    return 0


OPTIONS_LIMITS = [
    flag("--memory", default="", metavar="SIZE",
         help="the most a model and its caches may use here -- 90G, 24576M, a plain "
              "number of bytes. Every preflight, fit and lease reads it"),
    flag("--servers", type=int, default=None, metavar="N",
         help="the most model servers to run at once"),
    flag("--slots", type=int, default=None, metavar="N",
         help="the most conversations one server may hold"),
    flag("--idle", default="", metavar="TIME",
         help="stop a server unused for this long -- 10m, 1h, 600. "
              "`ml-stack-serve reclaim` and the fleet daemon act on it"),
    flag("--clear", action="store_true", help="take every limit off this machine"),
]


def cmd_limits(args: argparse.Namespace) -> int:
    """``ml-stack-serve limits`` -- how much of this machine ml-stack may take.

    Set nothing and it prints what is set; every limit is off until somebody sets one.
    """
    try:
        limits = ops.limits(memory_size=args.memory, servers=args.servers,
                            slots=args.slots, idle=args.idle, clear=bool(args.clear))
    except Refused as no:
        warn(f"error: {no.lines[0]}")
        return 2

    if args.clear:
        say(f"every limit is off; {limits.where} says so")
        return 0

    if limits.lines:
        say(f"what ml-stack may take here ({limits.where}):")
        for line in limits.lines:
            say(f"  {line}")
    else:
        say("nothing is limited here; ml-stack may use whatever this machine allows")
    if limits.machine:
        say(f"\nthis machine allows {human_bytes(limits.machine)}; a model may use "
            f"{human_bytes(limits.room)}")
    return 0


OPTIONS_RECLAIM = [
    flag("--idle", default="", metavar="TIME",
         help="how long unused is idle (default: what `limits --idle` set)"),
    flag("--settle", default="30", metavar="TIME",
         help="how long to watch before deciding, on top of what earlier looks found "
              "(default: %(default)ss)"),
    flag("--watch", action="store_true",
         help="keep looking rather than making one pass"),
    flag("--every", default="60", metavar="TIME",
         help="with --watch: how often to look (default: %(default)ss)"),
]


def cmd_reclaim(args: argparse.Namespace) -> int:
    """``ml-stack-serve reclaim`` -- stop the servers nobody is using.

    Idleness is asked of each server and what the looks found is kept, so a pass adds
    `--settle` seconds of its own watching to whatever the daemon has already seen.
    """
    from ml_stack.bench.history import parse_duration
    from ml_stack.serve.reclaim import watching

    try:
        older, watcher = ops.reclaim(idle=args.idle)
    except Refused as no:
        warn(no.lines[0])
        return 2
    if args.watch:
        every = parse_duration(args.every) or 60.0
        say(f"watching every {every:.0f}s, reclaiming after {older:.0f}s idle; "
            "Ctrl-C to stop", flush=True)
        try:
            with watching(older_than=older, every=every, idleness=watcher, say=print):
                while True:
                    time.sleep(3600)
        except KeyboardInterrupt:
            return 0
    stopped = ops.reclaim_once(older, watcher, settle=parse_duration(args.settle) or 0.0)
    if not stopped:
        say("nothing has been idle that long")
    return 0
