"""``ml-stack-workspace agent start|stop|list``: a local model that joins the workspace."""

from __future__ import annotations

import argparse
import time
from typing import Any

from ml_stack import roles
from ml_stack.command import flag
from ml_stack.log import say, warn
from ml_stack.sentinel import human
from ml_stack.serve import provenance
from ml_stack.workspace import (
    backlog,
    issuepump,
    localagent as la,
    localeffort as le,
    localloop,
    localmodel,
    localprofile as lp,
    localstart as ls,
    plain,
    task_launch,
    task_scheduler,
    tokens,
)
from ml_stack.workspace.service import Workspace

__all__ = ["ACTIONS", "OPTIONS", "run"]

ACTIONS = ("start", "stop", "list", "backlog", "schedule", "supersede-issue", "resume-issue")
READY_WAIT_S = 180.0
OPTIONS = [
    flag("action", choices=ACTIONS, help="start a local model as an agent, stop one, or list them"),
    flag("target", nargs="?", default="", metavar="NAME", help="stop: the agent to stop"),
    flag("--task", default="", help="existing assigned canonical task authorizing worker management"),
    flag("--model", default=localmodel.AUTO, metavar="auto|ID",
         help="auto: the best downloaded Qwen model that fits this machine, or the id "
              "of a downloaded model"),
    flag("--name", default="", help="the agent's workspace name (default: local- and the model's short name)"),
    flag("--role", default="", choices=list(roles.ROLES),
         help="what it may do; coding defaults to plan-and-go, chat to the standard role"),
    flag("--project", default="", metavar="PATH", help="the project folder it works for"),
    flag("--repo", default="", metavar="OWNER/REPO", help="project issue source; defaults to the project's GitHub origin"),
    flag("--issue", type=int, default=0, help="supersede-issue: obsolete repository issue number"),
    flag("--reason", default="", help="supersede-issue: current owner decision superseding the issue"),
    flag("--effort", default=le.DEFAULT, choices=[*le.LEVELS, le.AUTO],
         help="how much the model thinks (off is fastest); auto picks per task; default: %(default)s"),
    flag("--max-effort", default=le.DEFAULT_MAX, choices=list(le.LEVELS),
         help="the most effort the model may give itself with set_effort; default: %(default)s"),
    flag("--max-output-tokens", type=int, default=8192,
         help="maximum generated tokens per response, independent of reasoning effort"),
    flag("--orders-from", default=",".join(la.DEFAULT_ORDERS_FROM), metavar="NAMES",
         help="agents it takes tasks from besides the person and any lead (comma list)"),
    flag("--profile", default="coding", choices=["chat", "coding"],
         help="coding: durable project task queue; chat: interactive message loop"),
    flag("--harness", default="pi", choices=["pi", "codex", "claude"], help="native coding harness"),
    flag("--ctx", default="", metavar="TOKENS", help="context to serve (profile default), or such as 32k, 128k or 256k"),
    flag("--for", dest="lease_for", default="", metavar="TEXT",
         help="why the model is leased, one line, shown by `ml-stack-serve status|leases|history`"),
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
    if args.agent:
        if not args.task or not (args.name or args.target):
            raise ValueError('agent start requires an existing worker name and --task')
        got = task_launch.start(ws, tokens.load(ws.base, args.agent), args.name or args.target, args.task)
        runner = la.load(ws, got.name)
        caps = localloop.caps_of(runner)
        say(f'{got.name}: {runner.model_name}; {runner.ctx} context; {runner.role}; {runner.harness}; '
            f'effort {runner.effort}, ceiling {runner.max_effort}; '
            f'{caps.rounds} turns, {caps.calls} tool calls, {caps.steps} model calls, {caps.seconds:g}s wall time')
        return 0 if args.no_wait else _wait(ws, got.name, READY_WAIT_S)
    selected_profile = lp.profile(args.profile)
    selected_context = lp.parse_ctx(args.ctx) or selected_profile.ctx
    project = args.project or ("." if selected_profile.name == "coding" else "")
    selected_role = args.role or (la.PLAN_AND_GO if selected_profile.name == "coding" else roles.DEFAULT)
    pick = localmodel.choose(args.model, selection=localmodel.Selection(
        coding=selected_profile.name == "coding", context=selected_context))
    if not pick.ok:
        warn(pick.problem)
        if pick.hint:
            say(f"fetch it with: {pick.hint}")
        return 1
    say(f"model: {pick.name} ({pick.note})")
    try:
        got = ls.start(ws, ls.Ask(args.model, args.name, selected_role, args.effort, args.max_effort,
                                  args.profile, selected_context, project, la.check_orders(args.orders_from.split(",")), args.harness,
                                  repo=args.repo, max_output_tokens=args.max_output_tokens), pick=pick,
                       authority=ls.Authority(parent_token=ls.launch_parent(ws, project)))
    except ls.Unavailable as err:
        warn(str(err))
        return 1
    if got.already:
        say(f"{got.name} is already running (pid {got.pid}) on {got.model} as {got.role}")
        return 0
    say("it takes tasks from: the person, any lead, " + ", ".join(la.check_orders(args.orders_from.split(",")))
        + " (and their delegates); everything else is information only")
    say(f"started {got.name} (pid {got.pid}) on {got.model} as {got.role}; "
        f"log: {la.log_file(ws, got.name)}")
    return 0 if args.no_wait else _wait(ws, got.name, READY_WAIT_S)


def _stop(args: argparse.Namespace, ws: Workspace) -> int:
    if not args.target:
        raise ValueError("agent stop needs the agent's name; `agent list` shows them")
    done = (task_launch.stop(ws, tokens.load(ws.base, args.agent), args.target, args.task)
            if args.agent else ls.stop(ws, args.target))
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
    if args.action == 'schedule':
        if not args.target or not args.agent:
            raise ValueError('agent schedule requires a worker name and --agent registered-parent')
        return task_scheduler.watch(ws, tokens.load(ws.base, args.agent), la.check_name(args.target), task=args.task)
    if args.action in ("supersede-issue", "resume-issue"):
        if not args.target:
            raise ValueError("supersede-issue needs the existing worker name")
        if args.agent:
            token = tokens.load(ws.base, args.agent)
        else:
            human.require_person("supersede a repository issue")
            token = tokens.read_file(tokens.directory(ws.base) / tokens.OWNER_FILE)
        decision = backlog.supersede if args.action == "supersede-issue" else backlog.resume
        decision(ws, token, args.target, args.issue, args.reason)
        say(f"issue {args.issue}: {args.action} decision recorded")
        return 0
    if not (args.agent and args.action in ("start", "stop")):
        human.require_person(f"{args.action} a local agent")
    if args.action == "backlog":
        if not args.target:
            raise ValueError("agent backlog needs the existing worker's name")
        token = tokens.read_file(tokens.directory(ws.base) / tokens.OWNER_FILE)
        worker = la.load(ws, args.target)
        project = args.project or (worker.project if worker else "")
        repo = args.repo or (backlog.repository(project) if project else "")
        backlog.configure(ws, token, args.target, repo, project)
        issuepump.start(ws, token, args.target)
        say(f"{args.target} is pulling open issues from {repo} into its canonical task queue")
        return 0
    provenance.told(args.lease_for)
    handler: Any = _start if args.action == "start" else _stop
    return int(handler(args, ws))
