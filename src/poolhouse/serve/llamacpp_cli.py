"""``poolhouse-serve llama-cpp status|update|rollback|pin|list|prune``: following upstream llama.cpp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poolhouse import net
from poolhouse.command import flag, option
from poolhouse.log import say, warn
from poolhouse.serve import llamacpp_state, llamacpp_status, llamacpp_update
from poolhouse.serve.build_paths import BuildFailed
from poolhouse.serve.llamacpp_upstream import Upstream

__all__ = ["OPTIONS", "cmd_llama_cpp"]

ACTIONS = ("status", "update", "rollback", "pin", "list", "prune")
NEEDS_APPROVAL = 3

OPTIONS = [
    flag("action", choices=ACTIONS, metavar="ACTION",
         help="status: the active llama-server and whether upstream is ahead; update: build "
              "upstream and switch to it once it passes the smoke test; rollback: back to the "
              "previous build; pin BUILD: stop tracking and use that build; list: the builds "
              "held; prune: delete older good builds after asking"),
    flag("build", nargs="?", default="", help="the build to pin (see 'list')"),
    flag("--ref", default="master", metavar="BRANCH_TAG_OR_SHA",
         help="update: what to build (default master)"),
    flag("--jobs", type=int, default=0, metavar="N", help="update: parallel compile jobs"),
    flag("--force", action="store_true", help="update: rebuild a commit that is already built"),
    flag("--model", default="", metavar="GGUF",
         help="update: the model the smoke test loads (default: the smallest local GGUF)"),
    flag("--off", action="store_true", help="pin: resume tracking upstream"),
    flag("--keep", type=int, default=llamacpp_state.KEEP, metavar="N",
         help="prune: good builds to keep (default %(default)s)"),
    flag("--offline", action="store_true", help="status: skip the upstream check"),
    option("yes", help="prune: do not ask before deleting"),
    option("json"),
]


def cmd_llama_cpp(args: argparse.Namespace, *, upstream: Upstream | None = None,
                  pipeline: net.Pipeline | None = None) -> int:
    try:
        return _ACTIONS[args.action](args, upstream, pipeline)
    except net.NeedsApproval as exc:
        if args.json:
            say(json.dumps({"state": "needs_approval", "host": exc.host,
                            "fix": f"poolhouse-security approve-host {exc.host}"}))
        else:
            warn(f"error: {exc}")
        return NEEDS_APPROVAL
    except (BuildFailed, LookupError) as exc:
        warn(f"error: {exc}")
        return 2


def _status(args, upstream, pipeline) -> int:
    got = llamacpp_status.gather(upstream, pipeline, check=not args.offline)
    if args.json:
        say(json.dumps(got, indent=2))
        return 0
    if got["problem"]:
        warn(got["problem"])
        return 1
    say(f"{got['binary'] or 'no llama-server found'}")
    say(f"  from      {got['origin']}" + (f"; pinned to {got['pinned']}" if got["pinned"] else ""))
    say(f"  version   {got['version'] or 'unknown'}")
    say(f"  build     {got['build'] if got['build'] is not None else '?'}   commit {got['commit'] or '?'}")
    up = got["upstream"]
    if up:
        say(f"  upstream  newest build {up.get('tag') or '?'}, stable {up.get('stable') or '?'}, "
                        f"master {(up.get('master') or '?')[:9]}")
    say(f"  newer upstream commit: {got['newer']}" + (f" ({got['upstream_note']})" if got["upstream_note"] else ""))
    return 0


def _update(args, upstream, pipeline) -> int:
    env = llamacpp_update.Env(jobs=args.jobs, say=say)
    if upstream is not None:
        env.upstream = upstream
    if pipeline is not None:
        env.pipeline = pipeline
    result = llamacpp_update.update(args.ref, env, force=args.force,
                                    model=Path(args.model).expanduser() if args.model else None)
    if args.json:
        say(json.dumps({"status": result.status, "build": result.build, "detail": result.detail,
                        "kept": str(result.kept) if result.kept else ""}))
    elif result.status == "failed":
        warn(f"{result.build} failed its smoke test and was not activated: {result.detail}")
        warn(f"kept for inspection at {result.kept}; the active build is unchanged")
    else:
        say(f"{result.build}: {result.detail}")
    return 1 if result.status == "failed" else 0


def _rollback(args, upstream, pipeline) -> int:
    say(f"active: {llamacpp_state.rollback().name}")
    return 0


def _pin(args, upstream, pipeline) -> int:
    if args.off:
        llamacpp_state.unpin()
        say("tracking upstream again")
        return 0
    if not args.build:
        raise LookupError("pin needs a build name (see 'list') or --off")
    say(f"pinned to {llamacpp_state.pin(args.build).name}")
    return 0


def _list(args, upstream, pipeline) -> int:
    state, here = llamacpp_state.load(), llamacpp_state.active()
    rows = llamacpp_state.builds()
    for item in rows:
        marks = [m for m, on in (("active", here and here.name == item.name),
                                 ("pinned", state["pinned"] == item.name),
                                 ("previous", state["previous"][-1:] == [item.name])) if on]
        smoke = "smoke ok" if item.good else ("tracked" if item.tracked else "not tracked")
        say(f"{item.name:28} {str(item.info.get('built_at', '?'))[:19]:19} {smoke:12} {' '.join(marks)}")
    failed = sorted(llamacpp_state.failed_dir().glob("*")) if llamacpp_state.failed_dir().is_dir() else []
    for kept in failed:
        say(f"failed/{kept.name}")
    if not rows:
        say("no builds; `poolhouse-serve llama-cpp update` makes one")
    return 0


def _prune(args, upstream, pipeline) -> int:
    doomed = llamacpp_state.prune_candidates(args.keep)
    if not doomed:
        say(f"nothing to prune: {args.keep} good builds are kept besides the active and previous ones")
        return 0
    for item in doomed:
        say(f"would delete {item.path}")
    if not args.yes and input("delete these builds? type 'yes': ").strip().lower() != "yes":
        say("nothing deleted")
        return 0
    for item in doomed:
        llamacpp_state.remove(item)
        say(f"deleted {item.name}")
    return 0


_ACTIONS = {"status": _status, "update": _update, "rollback": _rollback, "pin": _pin,
            "list": _list, "prune": _prune}
