"""The setup, doctor and hello commands of the workspace CLI."""

from __future__ import annotations

import argparse

from poolhouse import agent_hooks
from poolhouse.log import say
from poolhouse.workspace import guide, onboard
from poolhouse.workspace.service import Workspace

__all__ = ["doctor", "hello", "setup"]


def setup(args: argparse.Namespace, ws: Workspace) -> int:
    for line in agent_hooks.install_report():
        say(line)
    if args.yes or args.rotate:
        done = onboard.setup(ws, args.agents or list(onboard.DEFAULT_AGENTS), args.rotate,
                             args.ttl_hours * 3600)
        for label, group in (("created", done.minted), ("kept", done.kept),
                             ("replaced", done.rotated), ("needs --rotate", done.lost)):
            if group:
                say(f"{label}: {', '.join(group)}")
        say(f"token files: {done.directory} (private, never printed)")
        for name in [*done.minted, *done.rotated, *done.kept]:
            say(f"\n--- paste into {name} ---\n{onboard.snippet(name)}")
        return 1 if any(not f.ok for f in onboard.doctor(ws)) else 0
    plan = guide.Plan(args.agents, 0.0 if args.no_live else args.live_seconds, args.wait_seconds)
    result = guide.walk(ws, plan)
    return 0 if not plan.live_s or not result["unconfirmed"] else 1


def doctor(args: argparse.Namespace, ws: Workspace) -> int:
    found = onboard.doctor(ws)
    for f in found:
        say(("ok   " if f.ok else "FIX  ") + f.what + ("" if f.ok else f"  -> {f.fix}"))
    return 0 if all(f.ok for f in found) else 1


def hello(args: argparse.Namespace, ws: Workspace) -> int:
    say(f"sent message {onboard.hello(ws, args.name)['seq']} to {args.name}")
    return 0
