"""``ml-stack-workspace agent start|stop|list``: a local model that joins the workspace."""

from __future__ import annotations

import argparse
import time
from typing import Any

from ml_stack import roles
from ml_stack.command import flag
from ml_stack.log import say, warn
from ml_stack.sentinel import human
from ml_stack.workspace import localagent as la, localmodel, localstart as ls, plain
from ml_stack.workspace.service import Workspace

__all__ = ["ACTIONS", "OPTIONS", "run"]

ACTIONS = ("start", "stop", "list")
READY_WAIT_S = 180.0
OPTIONS = [
    flag("action", choices=ACTIONS, help="start a local model as an agent, stop one, or list them"),
    flag("target", nargs="?", default="", metavar="NAME", help="stop: the agent to stop"),
    flag("--model", default=localmodel.AUTO, metavar="auto|ID",
         help="auto: the best downloaded mixture-of-experts Qwen model for agent work, or the id "
              "of a downloaded model"),
    flag("--name", default="", help="the agent's workspace name (default: local- and the model's short name)"),
    flag("--role", default=roles.DEFAULT, choices=list(roles.ROLES),
         help="what it may do (`/role` in ml-stack-chat describes them); default: %(default)s"),
    flag("--project", default="", metavar="PATH", help="the project folder it works for"),
    flag("--think", default="off", choices=("off", "on"), help="let the model think before it answers"),
    flag("--orders-from", default=",".join(la.DEFAULT_ORDERS_FROM), metavar="NAMES",
         help="agents it takes tasks from besides the person and any lead (comma list)"),
    flag("--no-wait", action="store_true", help="return as soon as the agent is started"),
]


def _wait(ws: Workspace, name: str, seconds: float) -> int:
    """Poll the status file until the agent is idle or failed; 0 ready, 1 failed or timed out."""
    began, shown = time.monotonic(), ""
    while time.monotonic() - began < seconds:
        row = next((r for r in ls.listing(ws) if r["name"] == name), None)
        if row is None or not row["running"]:
            warn(f"{name} stopped before it was ready; see {la.log_file(ws, name)}")
            return 1
        line = f"  {row['state']}: {row['detail']}".rstrip(": ")
        if line != shown:
            say(line)
            shown = line
        if row["state"] == "idle":
            say(f"{name} is ready: send it a task with `ml-stack-workspace send {name} task TEXT`")
            return 0
        if row["state"] == "failed":
            warn(f"{name} could not load its model: {row['detail']}")
            return 1
        time.sleep(1.0)
    say(f"{name} is still loading; `ml-stack-workspace agent list` shows when it is ready")
    return 0


def _start(args: argparse.Namespace, ws: Workspace) -> int:
    pick = localmodel.choose(args.model)
    if not pick.ok:
        warn(pick.problem)
        if pick.hint:
            say(f"fetch it with: {pick.hint}")
        return 1
    say(f"model: {pick.name} ({pick.note})")
    try:
        got = ls.start(ws, ls.Ask(args.model, args.name, args.role, args.think == "on", args.project,
                                  la.check_orders(args.orders_from.split(","))), pick=pick)
    except ls.Unavailable as err:
        warn(str(err))
        return 1
    if got.already:
        say(f"{got.name} is already running (pid {got.pid}) on {got.model} as {got.role}")
        return 0
    say(f"started {got.name} (pid {got.pid}) on {got.model} as {got.role}; "
        f"log: {la.log_file(ws, got.name)}")
    return 0 if args.no_wait else _wait(ws, got.name, READY_WAIT_S)


def _stop(args: argparse.Namespace, ws: Workspace) -> int:
    if not args.target:
        raise ValueError("agent stop needs the agent's name; `agent list` shows them")
    done = ls.stop(ws, args.target)
    say(f"stopped {done.name}" + (" (it had to be killed)" if done.forced else "")
        + ("; its model lease is released" if done.lease_released else ""))
    for note in done.notes:
        warn(note)
    return 0


def _list(ws: Workspace) -> int:
    rows = ls.listing(ws)
    if not rows:
        say("no local agent is running; `ml-stack-workspace agent start` starts one")
        return 0
    for r in rows:
        last = r["last_message"]
        heard = f", last from {plain.line(last.get('from', ''), 40)}" if last else ""
        say(f"{plain.line(r['name'], 48)}  {r['state']}  {plain.line(r['model'], 40)}  {r['role']}  "
            f"{r['tasks']} tasks, {r['steps']} steps{heard}"
            + (f"  {plain.line(r['detail'], 120)}" if r["detail"] else ""))
    return 0


def run(args: argparse.Namespace, ws: Workspace) -> int:
    """Do the ``agent`` action; start and stop are for a person at a terminal."""
    if args.action == "list":
        return _list(ws)
    human.require_person(f"{args.action} a local agent")
    handler: Any = _start if args.action == "start" else _stop
    return int(handler(args, ws))
