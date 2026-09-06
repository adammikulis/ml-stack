"""``ml-stack-walk``: open a page, press through its screens, and read what came back.

    ml-stack-walk fleet
    ml-stack-walk fleet cluster chat models --out ./shots
    ml-stack-walk graph search detail 3d --base http://127.0.0.1:8794

Each screen is screenshotted into ``--out`` and its text printed. The exit code is 1 when
any screen raised a console error or could not be reached, and 2 when the page never
answered. Both subcommands take the screens to walk and default to all of them; the work
is `ml_stack.walk.ops`, and the handlers here parse and print.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ml_stack import home
from ml_stack.command import Group, Option, flag, option
from ml_stack.fleet.daemon import DEFAULT_PORT as FLEET_PORT
from ml_stack.graph.serve import PORT as GRAPH_PORT
from ml_stack.log import say, warn
from ml_stack.walk import ops
from ml_stack.walk.ops import FLEET, GRAPH, Stop, Walk, WalkFailed

__all__ = ["COMMANDS", "main", "report"]

COMMANDS = Group(
    "ml-stack-walk",
    "Open the interface the way a person does: click through its screens, screenshot "
    "each one, print what it said, and refuse the walk when the console went red.")
main = COMMANDS.run

FLEET_BASE = f"http://127.0.0.1:{FLEET_PORT}"
GRAPH_BASE = f"http://127.0.0.1:{GRAPH_PORT}"


def _shared(base: str) -> list[Option]:
    """The options both walks take."""
    return [
        flag("screens", nargs="*", metavar="SCREEN",
             help="which screens to walk (default: all of them)"),
        flag("--base", default=base, help=f"where the page is served (default: {base})"),
        option("out", help="directory for the screenshots "
                           "(default: ~/.ml-stack/cache/walks/<page>)"),
        flag("--headless", action=argparse.BooleanOptionalAction, default=True,
             help="drive an invisible browser (default); --no-headless opens a window "
                  "where ML_STACK_WINDOW_POSITION says and gives the screen back"),
        option("timeout", default=20.0, help="seconds to wait for a screen (default: 20)"),
        flag("--chars", type=int, default=900,
             help="how much of each screen to print (default: 900)"),
    ]


def report(stops: Sequence[Stop], out: Path, chars: int) -> int:
    """Print every stop and return the exit code: 1 when any screen errored."""
    for stop in stops:
        if stop.skipped:
            say(f"── {stop.screen} ── skipped: {stop.skipped}", flush=True)
            continue
        say(f"── {stop.screen} ──" + (f" {stop.shot.name}" if stop.shot else ""),
            flush=True)
        if stop.text:
            say(stop.text[:chars] + ("…" if len(stop.text) > chars else ""), flush=True)
        for line in stop.errors:
            warn(f"   ! {line}", flush=True)
    walked = [s for s in stops if not s.skipped]
    errored = [s for s in stops if s.errors]
    say(f"{len(walked)} screen(s) walked, {len(stops) - len(walked)} skipped, "
        f"{len(errored)} with errors -> {out}")
    return 1 if errored else 0


def _walked(args: argparse.Namespace, page: str, **extra: object) -> int:
    """Build the walk from parsed arguments, run it and print it."""
    out = home.expand(args.out) if args.out else home.cache("walks", page)
    try:
        screens = ops.screens_of(page, args.screens)
        stops = ops.walk(Walk(page=page, base=args.base, out=out, screens=screens,
                              headless=args.headless, timeout_s=float(args.timeout),
                              **extra))  # type: ignore[arg-type]
    except (ValueError, WalkFailed) as exc:
        warn(f"ml-stack-walk: {exc}")
        return 2
    return report(stops, out, int(args.chars))


@COMMANDS.command(
    "fleet", help="the daemon's screens: the first-run steps, then cluster, chat, "
                  "models, settings and fit",
    epilog="screens: " + ", ".join(FLEET),
    options=[*_shared(FLEET_BASE),
             flag("--passphrase", default="",
                  help="the cluster's passphrase, for a daemon that asks for one"),
             flag("--setup", action="store_true",
                  help="press the first-run buttons that save this machine's "
                       "preferences; without it the wizard is walked only as far as it "
                       "goes without changing anything")])
def cmd_fleet(args: argparse.Namespace) -> int:
    return _walked(args, "fleet", passphrase=args.passphrase, setup=bool(args.setup))


@COMMANDS.command(
    "graph", help="the rendered graph page: the graph, the legend, the filter, an "
                  "entry, the map, 3D, the history, the question box, the review queue",
    epilog="screens: " + ", ".join(GRAPH),
    options=[*_shared(GRAPH_BASE),
             flag("--find", default="", help="what to type into the filter box"),
             flag("--ask", default="",
                  help="a question to send to the model behind the page; without it the "
                       "question box is only read")])
def cmd_graph(args: argparse.Namespace) -> int:
    return _walked(args, "graph", find=args.find, ask=args.ask)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
