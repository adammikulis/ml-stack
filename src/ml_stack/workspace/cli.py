"""``ml-stack-workspace``: the message bus, notes, scratch folders and claims from a shell."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ml_stack.command import Group, flag, option
from ml_stack.http import ServerError
from ml_stack.log import say, warn
from ml_stack.sentinel import human
from ml_stack.sentinel.human import HumanRequired
from ml_stack.workspace import (
    backlog,
    chat,
    coordinator_client,
    coordinator_config,
    filecli,
    guide,
    harness_remote,
    limits,
    localcli,
    localroute,
    nudge,
    onboard,
    project,
    project_connection,
    remote_cli,
    remote_task_client,
    task_integration,
    task_outcomes,
    task_worktree_recovery,
    tokens,
    worktree_lifecycle,
)
from ml_stack.workspace.boardapi import Follow
from ml_stack.workspace.boards import ANNOUNCE_KINDS, MODES, STYPES
from ml_stack.workspace.bus import CALL_TYPES, TYPES
from ml_stack.workspace.chain import ChainBroken
from ml_stack.workspace.claims import KINDS as CLAIM_KINDS, Conflict
from ml_stack.workspace.coordination import workspace_id
from ml_stack.workspace.identity import AGENT_MARKERS, ROLES, TOKEN_ENV, Denied
from ml_stack.workspace.modelid import describe
from ml_stack.workspace.notes import KINDS as NOTE_KINDS
from ml_stack.workspace.rates import RateLimited
from ml_stack.workspace.screen import Refused, fence
from ml_stack.workspace.service import Workspace
from ml_stack.workspace.taskboard import TaskBoard

__all__ = ["COMMANDS", "main"]

CANCELLED = threading.Event()
LABEL_ENV = "ML_STACK_WORKSPACE_LABEL"
CODES = ((ServerError, 3), (Denied, 3), (Refused, 3), (RateLimited, 4), (Conflict, 5), (ChainBroken, 6),
         (HumanRequired, 3), (EOFError, 2), (ValueError, 2), (OSError, 2))
Handler = Callable[[argparse.Namespace, Workspace, str], Any]

COMMON = [option("json"),
          flag("--request-id", default="", help="reuse an exact remote mutation request after a lost response"),
          flag("--token-file", default="", help=f"a file holding the sender's token "
                                                f"(else --agent, else ${TOKEN_ENV}, "
                                                f"else ${tokens.AGENT_ENV})"),
          flag("--agent", default="", help="act as this agent: its token file from `join`"),
          flag("--label", default="", help=f"a note of which helper is acting, shown as "
                                           f"NAME/LABEL (else ${LABEL_ENV}); not an authority")]
WIDEN = [flag("--limit", type=int, default=0,
              help="show this many (default: a few, each cut short; the rest is counted)"),
         flag("--all", action="store_true", help="show everything, uncut")]
READ = [*WIDEN, flag("--ack", action="store_true", help="mark what is shown as read"),
        flag("--raw", action="store_true", help="also show the unfenced text of clear messages")]
CLAIM = [flag("kind", choices=CLAIM_KINDS), flag("key")]
OWNER = flag("--owner", default="", help="a lead or human may name another agent")


def _label(args: argparse.Namespace) -> str:
    return args.label or os.environ.get(LABEL_ENV, "")


def _token(args: argparse.Namespace) -> str:
    connection = _project_connection()
    if connection is not None:
        return _context(args, connection)[1]
    base = limits.root()
    remote = coordinator_client.client(base)
    return _coordinator_token(args, remote) if remote else _local_token(args)


def _coordinator_token(args: argparse.Namespace, remote) -> str:
    base = limits.root()
    agent = args.agent or os.environ.get(tokens.AGENT_ENV, "")
    if agent and not args.token_file:
        args.agent = remote.ensure(base, agent, project=project.authoritative())
    return tokens.resolve(base, token_file=args.token_file, agent=args.agent)


def _local_token(args: argparse.Namespace) -> str:
    base = limits.root()
    agent = args.agent or os.environ.get(tokens.AGENT_ENV, "")
    if agent and not args.token_file:
        ws = Workspace(base)
        try:
            ws.auth(tokens.load(base, agent))
        except Denied:
            guide.agent_connect(ws, agent, project.describe())
    return tokens.resolve(base, token_file=args.token_file, agent=args.agent)


def _project_connection(cwd: Path | None = None):
    return project_connection.selected(cwd) or project_connection.auto_attach(cwd)


_CONNECTION_UNSET = object()


def _context(args: argparse.Namespace, connection=_CONNECTION_UNSET):
    if connection is _CONNECTION_UNSET:
        connection = _project_connection()
    if connection is None:
        return Workspace(), _local_token(args)
    remote = project_connection.RemoteWorkspace(connection["host"], connection["project_id"],
                                                cluster=connection.get("cluster", ""),
                                                cluster_key=Path(connection["cluster_key"])
                                                if connection.get("cluster_key") else None)
    token = remote.token(agent=args.agent or os.environ.get(tokens.AGENT_ENV, "") or connection.get("agent", ""),
                         token_file=getattr(args, "token_file", ""))
    if not connection.get("agent"):
        who = remote.call("whoami", token)
        project_connection.bind(remote, Path(connection["root"]), who["id"], connection.get("cluster", ""))
    return project_connection.CanonicalWorkspace(remote, token), token


def _block(lines: list[str], what: str) -> str:
    return fence("\n".join(lines), f"workspace:{what}", "names and subjects written by agents").text


def _row(value: dict[str, Any]) -> str:
    if "root" in value:
        return (f"[{value['root']}] {value['subject']}  ({value['from']}, {value['replies']} "
                f"replies, {value['unread']} unread)")
    if "members" in value:
        return (f"{value['name']}  {value['unread']} unread, {value['posts']} posts"
                f"{'' if value['member'] else ', not a member'}  {value['title']}")
    if "mode" in value:
        return f"{value['type']} {value['target']} -> {value['mode']}".replace("  ", " ")
    return f"{value['a']} <-> {value['b']}  {value['messages']} messages, {value['unread']} unread"


def _text(value: Any) -> str:
    if isinstance(value, dict) and "text" in value and "seq" in value:
        where = f" on {value['board']}" if value.get("board") else ""
        model = ("" if value.get("from_role") == "human"
                 else f" ({describe(value.get('from_model', ''), value.get('from_model_state', ''))})")
        return (f"[{value['seq']}] {value['type']} from {value.get('from_label', value['from'])}"
                f"{model}{where} ({value['trust']}, no authority, {value['state']})\n{value['text']}")
    if isinstance(value, dict) and {"block", "uses"} <= value.keys():
        return str(value["block"]).rstrip("\n")
    if isinstance(value, dict) and value.get("authority") == "none" and "text" in value:
        return str(value["text"])
    if isinstance(value, dict) and "handle" in value and "line" in value and "text" in value:
        return str(value["text"])
    if isinstance(value, list) and value and all(
            isinstance(v, dict) and ({"root", "replies"} <= v.keys() or {"members", "posts"} <= v.keys()
                                     or {"a", "b", "messages"} <= v.keys()) for v in value):
        return _block([_row(v) for v in value], "board")
    if isinstance(value, list) and value and all(
            isinstance(v, dict) and {"type", "target", "mode"} == v.keys() for v in value):
        return _block([_row(v) for v in value], "subscriptions")
    if isinstance(value, list) and value and all(
            isinstance(v, dict) and {"id", "role", "model_state", "last_acted"} <= v.keys() for v in value):
        return _block([f"{v['id']}  {v['role']}{'  child of ' + v['parent'] if v['parent'] else ''}  {describe(v['model'], v['model_state'])}"
                       f"{'  ' + v['harness'] if v['harness'] else ''}" for v in value], "agents")
    if isinstance(value, dict) and {"kind", "key", "owner", "expires_in_s"} <= value.keys():
        soon = ", expiring soon" if value.get("expiring_soon") else ""
        return f"{value['kind']} {value['key']}  {value['owner']}  expires in {value['expires_in_s']:.0f} s{soon}"
    if isinstance(value, dict) and "text" in value and "kind" in value:
        return (f"note {value['id']} {value['kind']} ({value['trust']}"
                f"{', stale' if value['stale'] else ''}): {value['status']}\n{value['text']}")
    if isinstance(value, list):
        return "\n".join(_text(v) for v in value) or "(none)"
    if isinstance(value, dict):
        return "\n".join(f"{k}: {v if not isinstance(v, (dict, list)) else json.dumps(v)}"
                         for k, v in value.items())
    return str(value)


def _show(args: argparse.Namespace, value: Any) -> None:
    say(json.dumps(value, sort_keys=True, default=str) if args.json else _text(value))


def _body(text: str) -> str:
    return sys.stdin.read() if text == "-" else text


def _ids(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _held_note(value: Any) -> None:
    """Say on stderr how many results the default caps held back, and how to see them."""
    held = getattr(value, "held", 0)
    if held:
        warn(f"workspace: {held} more held back (not shown, still unread); "
             f"use --limit N or --all to see more")


def _watch(args: argparse.Namespace, ws: Workspace, token: str) -> int:
    """Print each batch of new messages as it arrives; 0 after one batch with --once, 3 when
    the timeout passes with nothing."""
    began = time.monotonic()
    scope = {k: v for k, v in (("board", args.board), ("thread", args.thread), ("dm", args.dm)) if v}
    if len(scope) > 1:
        raise ValueError("watch one of --board, --thread or --dm")
    spec = Follow(**scope, after=args.since, timeout_s=5.0)
    while True:
        left = args.timeout - (time.monotonic() - began) if args.timeout else 5.0
        if args.timeout and left <= 0:
            return 3
        if scope:
            got = ws.follow(token, replace(spec, timeout_s=min(5.0, left)), CANCELLED.is_set)
            spec, batch = replace(spec, after=got["seq"]), got["messages"]
        else:
            batch = ws.wait(token, min(5.0, left), ack=True, cancel=CANCELLED.is_set,
                            limit=args.limit, widen=args.all)
        for item in batch:
            _show(args, item)
            sys.stdout.flush()
        _held_note(batch)
        if batch and args.once:
            return 0


def _mint(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    return {"token": ws.mint(token, args.name, args.role, args.ttl_hours * 3600)}


def _revoke(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    return {"revoked": ws.revoke(token, args.name, args.tree)}


def _invite(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    made = ws.invite(token, args.name, _ttl(args.ttl), args.uses)
    return {"block": onboard.snippet("", made["code"], args.name, made["project"],
                                     (made["uses"], int(made["ttl_s"] // 60))), "uses": made["uses"]}


def _whoami(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    who = ws.auth(token)
    if args.model:
        ws.claim_model(token, args.model, args.harness)
    model, state = ws.model_of(who.id)
    return {"id": who.id, "role": who.role, "project": ws.registry.info(who.id)["project"],
            "model": model or "unknown", "model_state": state, "harness": ws.registry.info(who.id)["harness"]}


def _hello_model(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    return ws.claim_model(token, args.model, "", args.label_name)


def _agents(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    ws.auth(token)
    return ws.registered()


def _send(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    return ws.send(token, args.to, args.type, _body(args.body), subject=args.subject,
                   reply_to=args.reply_to, ttl_s=args.ttl, label=_label(args))


def _inbox(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    if not args.children:
        roll = None if args.json else ws.board.rollup(token, args.ack)
        if roll:
            say(_text(roll))
        return ws.inbox(token, args.ack, args.limit, args.raw, args.all)
    if args.ack:
        raise ValueError("--children shows some of the unread messages, so it cannot --ack")
    me = ws.auth(token).id
    return [m for m in ws.inbox(token, False, 0, args.raw, True) if m["from"].startswith(me + "/")]


def _wait(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    return ws.wait(token, args.timeout, args.ack, args.raw, limit=args.limit, widen=args.all)


def _announce(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    return ws.announce(token, args.kind, _body(args.text), _label(args))


def _note_add(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    return ws.note_add(token, args.kind, args.title, _body(args.body), source=args.source,
                       tags=[t for t in args.tags.split(",") if t],
                       supersedes=_ids(args.supersedes), verify_cmd=args.verify_cmd,
                       ttl_s=args.ttl_days * 86400)


def _claim(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    note = f"[{_label(args)}] {args.note}".strip() if _label(args) else args.note
    return ws.claim(token, args.kind, args.key, ttl_s=args.ttl, pid=args.pid, note=note, label=_label(args))


def _worktrees(args, ws, token):
    return worktree_lifecycle.pending(ws.base, ws.auth(token).id, _label(args))


def _released(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    return {"released": args.qid, "text_chars": len(ws.quarantine_release(token, args.qid))}


def _init(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    if not any(os.environ.get(m) for m in AGENT_MARKERS) and not (
            sys.stdin.isatty() and sys.stdout.isatty()):
        raise Denied("init needs a person at a terminal on stdin and stdout")
    return {"token": ws.init(args.name), "note": f"keep this; set {TOKEN_ENV} to use it"}


def _setup(args: argparse.Namespace, ws: Workspace) -> int:
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


def _connect(args: argparse.Namespace, ws: Workspace) -> int:
    agent = args.agent or os.environ.get(tokens.AGENT_ENV, "")
    if agent:
        if args.no_project or args.one_agent or args.remote or args.code_only or args.name:
            raise Denied("agent connect takes --agent and --project; invite options need a person")
        connection = _project_connection(Path(args.project) if args.project else None)
        if connection is not None:
            canonical, token = _context(args, connection)
            who = canonical.auth(token)
            info = canonical.registry.info(who.id)
            if info.get("project", {}).get("key") != connection["project_id"]:
                raise Denied("this identity is not authorized for the canonical project")
            _show(args, {"id": who.id, "project": connection["project_id"], "state": "connected"})
            return 0
        found = project.describe(args.project)
        remote = coordinator_client.client(ws.base)
        if remote:
            found = project.authoritative(args.project)
            name = remote.ensure(ws.base, agent, project=found)
            result = {"id": name, "project": found.get("name", ""), "state": "connected"}
        else:
            result = guide.agent_connect(ws, agent, found)
        _show(args, result)
        return 0
    plan = guide.Plan([args.name] if args.name else [], 0.0 if args.no_live else args.live_seconds,
                      args.wait_seconds, shared=not args.one_agent,
                      project=project.describe(args.project, none=args.no_project),
                      code_only=getattr(args, "code_only", False), remote=getattr(args, "remote", False))
    answered = guide.connect(ws, plan)
    return 0 if answered or plan.live_s == 0 or plan.code_only or plan.wait_s <= 0 else 1


def _join(args: argparse.Namespace, ws: Workspace) -> int:
    if args.coordinator:
        coordinator_client.connect(limits.root(), args.coordinator)
    remote = coordinator_client.client(limits.root())
    expected = getattr(args, "workspace", "")
    if expected and (not remote or remote.config["workspace"] != expected):
        raise Denied("this invitation belongs to another coordinator workspace; "
                     "enroll in its Fleet cluster and use the complete invitation command")
    name = remote.join(limits.root(), args.code, args.name, args.model, args.harness) if remote else onboard.join(
        ws, args.code, args.name, claim=(args.model, args.harness))
    if not remote:
        ws.registry._record_device(name, onboard.device_metadata.current())
    say(f"joined as {name}")
    return 0


def _doctor(args: argparse.Namespace, ws: Workspace) -> int:
    found = onboard.doctor(ws)
    for f in found:
        say(("ok   " if f.ok else "FIX  ") + f.what + ("" if f.ok else f"  -> {f.fix}"))
    return 0 if all(f.ok for f in found) else 1


def _chat(args: argparse.Namespace, ws: Workspace) -> int:
    human.require_person("workspace chat")
    token = (tokens.read_file(Path(args.token_file).expanduser()) if args.token_file
             else tokens.read_file(tokens.directory(ws.base) / tokens.OWNER_FILE))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: CANCELLED.set())
    chat.run(ws, token, Follow(board=args.board, dm=args.to, backlog=args.backlog),
             chat.Console(sys.stdin, say, CANCELLED))
    return 0


def _hello(args: argparse.Namespace, ws: Workspace) -> int:
    say(f"sent message {onboard.hello(ws, args.name)['seq']} to {args.name}")
    return 0


def _brief(args: argparse.Namespace, ws: Workspace) -> int:
    me = args.agent or os.environ.get(tokens.AGENT_ENV, "")
    if not me:
        raise ValueError("name the parent with --agent NAME (or set ML_STACK_WORKSPACE_AGENT)")
    say(onboard.brief(args.name, me), end="")
    return 0


def _heartbeat(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    claims = ws.renew(token, args.ttl)
    return {"renewed": len(claims), "capped": [f"{c['kind']}:{c['key']}" for c in claims if c["capped"]]}


BOARD_ACTIONS = ("list", "read", "post", "threads", "create", "add", "mentions")


def _board(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    b, name, rest = ws.board, args.name, args.rest
    if args.action == "list":
        return b.list(token)
    if args.action == "mentions":
        return b.mentions(token, args.limit or 50)
    if not name:
        raise ValueError(f"board {args.action} needs a board name such as #general")
    if args.action == "read":
        return b.read(token, name, args.limit, args.after, not args.no_mark)
    if args.action == "threads":
        return b.threads(token, name, args.limit or 50)
    if args.action == "create":
        return b.create(token, name, private=args.private, title=" ".join(rest))
    if not rest:
        raise ValueError(f"board {args.action} needs one more argument")
    if args.action == "add":
        return b.add(token, name, rest[0])
    return ws.send(token, name, args.type, _body(" ".join(rest)), subject=args.subject,
                   reply_to=args.reply_to, label=_label(args))


def _dm(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    if not args.name:
        return ws.board.dm_list(token)
    if args.body:
        ws.send(token, args.name, args.type, _body(args.body), subject=args.subject,
                reply_to=args.reply_to, label=_label(args))
    return ws.board.dm(token, args.name, args.between, args.limit)


def _subscribe(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    return ws.board.subscribe(token, args.type, args.target, args.mode, args.force)


def _digest(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    return ws.board.digest(token, args.ack, args.thread)


def _ttl(text: str) -> float:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return float(text[:-1]) * units[text[-1]] if text and text[-1] in units else float(text or 0)


def _hook_snippet(args: argparse.Namespace, ws: Workspace) -> int:
    say(onboard.hook_snippet(args.tool, args.agent or "NAME"), end="")
    return 0


def _nudging(args: argparse.Namespace) -> int:
    if coordinator_config.load(limits.root()).get("mode") == "remote":
        raise Denied("this watcher is local-only; use coordinator inbox polling")
    if args.hook:
        return _hook(args)
    ws, token = _context(args)
    line = ws.nudge(token)
    if line:
        say(line)
    return 0


def _hook(args: argparse.Namespace) -> int:
    if coordinator_config.load(limits.root()).get("mode") == "remote":
        raise Denied("this watcher is local-only; use coordinator inbox polling")
    stdin = sys.stdin.read() if args.hook == "stop" and not sys.stdin.isatty() else ""
    try:
        ws, token = _context(args)
        out = nudge.output(args.hook, ws.waiting(token), stdin)
    except tuple(kind for kind, _ in CODES):
        return 0
    if out:
        say(out)
    return 0


def _install_hooks(args: argparse.Namespace, ws: Workspace) -> int:
    path = Path(args.settings).expanduser()
    events = onboard.install_hooks(path, args.agent or "claude-code")
    say(f"wrote {', '.join(events)} to {path}")
    return 0


def _board_serve(args: argparse.Namespace, ws: Workspace) -> int:
    listener = localroute.serve(ws, args.port)
    say(f"the Board, read-only, for the person: http://127.0.0.1:{listener.port}/")
    say(f"the Agents panel (start and stop need this browser session): "
        f"http://127.0.0.1:{listener.port}/agents?session={listener.session}")
    listener.start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        listener.stop()
    return 0


LIVE = [flag("--no-live", action="store_true", help="skip the live check"),
        flag("--live-seconds", type=float, default=120.0, help="how long the live check waits"),
        flag("--wait-seconds", type=float, default=600.0, help="how long to wait for a join")]
BARE: tuple[tuple[str, str, list[Any], Callable[[argparse.Namespace, Workspace], int]], ...] = (
    ("connect", "initialize or reconnect --agent ID; without an agent, share a person-approved invite",
     [flag("--name", default="", help="a suggested id for the agent; it may pick its own"),
      flag("--project", default="", help="the project folder (default: the git root you are in)"),
      flag("--no-project", action="store_true", help="connect without naming a project"),
      flag("--one-agent", action="store_true",
           help="a single-use code (default: one paste for up to 10 agents, one hour)"),
      flag("--remote", action="store_true", help="require an active shared host and include its authority"),
      flag("--code-only", action="store_true", help="print and copy the invite, then exit without waiting"),
      flag("--label", default="", help="the helper label under this agent identity"),
      *LIVE], _connect),
    ("join", "an agent redeems an invite code and saves its private token", [
        flag("code"), flag("--name", default="", help="a short id for yourself, e.g. codex"),
        flag("--coordinator", default="", help="select this enrolled Fleet coordinator before redeeming the invite"),
        flag("--workspace", default="", help="expected coordinator workspace ID; refuses local redemption"),
        flag("--model", default="", help="the exact model id you run as; recorded as claimed"),
        flag("--harness", default="", help="your harness, e.g. claude-code or codex")],
     _join),
    ("setup", "guided walkthrough for several agents; --yes makes token files directly", [
        flag("agents", nargs="*", help="suggested ids (--yes: the agents to create)"),
        flag("--rotate", action="append", default=[], metavar="NAME",
             help="replace this agent's token (repeatable)"),
        flag("--ttl-hours", type=float, default=720.0, help="how long new tokens last"),
        flag("--yes", action="store_true", help="no questions: token files for the names given"),
        *LIVE], _setup),
    ("board-serve", "serve the read-only Board page on a loopback port; the person's identity, no token in the page",
     [flag("--port", type=int, default=0)], _board_serve),
    ("agent", "start a local model as an agent that takes and gives tasks, stop one, or list them; start and stop at a terminal",
     localcli.OPTIONS, lambda a, w: localcli.run(a, w)),
    ("chat", "live conversation with a board or one agent: streams it, sends each line you type; at a terminal",
     [flag("--board", default="", help="a board such as #general"),
      flag("--to", default="", help="an agent id"), flag("--backlog", type=int, default=20),
      flag("--token-file", default="", help="the person's token file (default: the owner file)")],
     _chat),
    ("doctor", "check the whole setup and say what to fix; at a terminal", [], _doctor),
    ("hello", "send AGENT the first message ('workspace ready'); at a terminal", [flag("name")],
     _hello),
    ("snippet", "print the paste block for AGENT (no secret in it)", [flag("name")],
     lambda a, w: say(onboard.snippet(a.name), end="") or 0),
    ("hook-snippet", "print the setting that makes a tool run `nudge` after each step; writes nothing",
     [flag("tool", choices=("claude-code", "codex"))], _hook_snippet),
    ("install-hooks", "write the nudge hooks (PostToolUse, Stop, UserPromptSubmit) into Claude Code's "
     "settings; at a terminal", [flag("--settings", default="~/.claude/settings.json",
                                      help="the Claude Code settings file")], _install_hooks),
    ("brief", "print the short brief a parent pastes into a subagent's prompt", [flag("name")],
     _brief),
)

TABLE: tuple[tuple[str, str, list[Any], Handler], ...] = (
    ("init", "register the first human identity, at a terminal", [flag("--name", default="owner")],
     _init),
    ("mint", "print a new token for an agent id", [
        flag("name"), flag("--role", choices=ROLES, default="agent"),
        flag("--ttl-hours", type=float, default=0.0)], _mint),
    ("revoke", "stop an agent's token working, with its outstanding invites; --tree also revokes every agent below it",
     [flag("name"), flag("--tree", action="store_true", help="revoke every descendant too")], _revoke),
    ("invite", "a joined agent makes a one-time paste block for a new agent it starts; the new agent joins as its child", [
        flag("--name", default="", help="a suggested id for the new agent"),
        flag("--ttl", default="10m", help="how long the code works, e.g. 10m (at most 30m)"),
        flag("--uses", type=int, default=1, help="how many agents may join with it (at most 3)")], _invite),
    ("whoami", "who the token says you are; --model records your own model id as claimed", [
        flag("--model", default="", help="the exact model id you run as (a label, never a right)"),
        flag("--harness", default="", help="your harness, e.g. claude-code or codex")], _whoami),
    ("hello-model", "record the model a helper LABEL of yours runs (claimed)", [
        flag("label_name", metavar="LABEL"), flag("model", metavar="MODEL")], _hello_model),
    ("agents", "every live identity with its role, model and whether the model is verified", [],
     _agents),
    ("send", "send a message (BODY - reads stdin)", [
        flag("to", help="an agent id, or * for the announcements board (joined, milestone, "
                        "done, blocked only)"), flag("type", choices=CALL_TYPES),
        flag("body"), flag("--subject", default=""), flag("--reply-to", type=int, default=0),
        flag("--ttl", type=float, default=0.0, help="seconds until it expires")], _send),
    ("task-create", "create a canonical task in your existing project grant", [flag("payload")],
     lambda a, w, t: TaskBoard(w).create(t, json.loads(_body(a.payload)))),
    ("tasks", "authorized canonical tasks and progress metrics", [], lambda a, w, t: TaskBoard(w).list(t)),
    ("task", "task lease, checkpoints, proposal and independent review", [flag("id")],
     lambda a, w, t: TaskBoard(w).get(t, a.id)),
    ("task-subscribe", "receive inbox status changes for a task", [flag("id")],
     lambda a, w, t: TaskBoard(w).subscribe(t, a.id)),
    ("task-unsubscribe", "stop inbox status changes for a task", [flag("id")],
     lambda a, w, t: TaskBoard(w).unsubscribe(t, a.id)),
    ("issue-subscribe", "receive inbox status changes for a GitHub issue", [flag("ref", metavar="OWNER/REPO#NUMBER")],
     lambda a, w, t: backlog.subscribe_issue(w, t, a.ref)),
    ("issue-unsubscribe", "stop inbox status changes for a GitHub issue", [flag("ref", metavar="OWNER/REPO#NUMBER")],
     lambda a, w, t: backlog.subscribe_issue(w, t, a.ref, False)),
    ("task-claim", "claim a queued task with an existing trusted allocation", [flag("id"), flag("allocation_id")],
     lambda a, w, t: TaskBoard(w).claim(t, a.id, a.allocation_id)),
    ("task-heartbeat", "renew your active canonical task lease", [flag("id")],
     lambda a, w, t: TaskBoard(w).heartbeat(t, a.id)),
    ("task-checkpoint", "save your active task checkpoint (JSON or - for stdin)", [flag("id"), flag("payload")],
     lambda a, w, t: TaskBoard(w).checkpoint(t, a.id, json.loads(_body(a.payload)))),
    ("task-submit", "submit immutable artifact hashes for review (JSON or -)", [flag("id"), flag("payload")],
     lambda a, w, t: TaskBoard(w).submit(t, a.id, json.loads(_body(a.payload)))),
    ("task-review", "independently review an authorized task (JSON or -)", [flag("id"), flag("payload")],
     lambda a, w, t: task_outcomes.review(w, t, a.id, json.loads(_body(a.payload)))),
    ("task-credit", "retry recording an authorized immutable outcome", [flag("id")],
     lambda a, w, t: task_outcomes.credit(w, t, a.id)),
    ("task-integrate", "gate, land and clean an independently accepted committed native task", [flag("id")],
     lambda a, w, t: task_integration.integrate(w, t, a.id)),
    ("task-dematerialize", "remove an unchanged inactive task checkout, preserving its pending history", [flag("id"), flag("reason")],
     lambda a, w, t: task_worktree_recovery.dematerialize(w, t, a.id, a.reason)),
    ("inbox", "unread messages, fenced as data", [
        *READ,
        flag("--children", action="store_true", help="only messages from your delegates")],
     _inbox),
    ("board", "boards: list, read NAME, post NAME TEXT, threads NAME, create NAME [TITLE], add NAME AGENT, mentions", [
        flag("action", choices=BOARD_ACTIONS), flag("name", nargs="?", default=""),
        flag("rest", nargs="*"), flag("--type", choices=TYPES, default="note"),
        flag("--subject", default=""), flag("--reply-to", type=int, default=0),
        flag("--limit", type=int, default=0, help="read: how many (default: a few, cut short)"),
        flag("--after", type=int, default=0),
        flag("--private", action="store_true", help="create: only people you add can join"),
        flag("--no-mark", action="store_true", help="read: leave the board's unread count")],
     _board),
    ("join-board", "join an open board", [flag("name")], lambda a, w, t: w.board.join(t, a.name)),
    ("leave-board", "leave a board", [flag("name")], lambda a, w, t: w.board.leave(t, a.name)),
    ("dm", "the two-sided conversation with NAME (BODY sends first); no NAME lists them", [
        flag("name", nargs="?", default=""), flag("body", nargs="?", default=""),
        flag("--type", choices=TYPES, default="note"), flag("--subject", default=""),
        flag("--reply-to", type=int, default=0), flag("--limit", type=int, default=50),
        flag("--between", default="", help="a person or lead: the pair BETWEEN and NAME")],
     _dm),
    ("subscribe", "choose what reaches your inbox: board, thread, agent, kind or mentions", [
        flag("type", choices=STYPES), flag("target", nargs="?", default=""),
        flag("--mode", choices=MODES, default="inbox",
             help="inbox, digest (summarised by `digest`) or silent (kept, never delivered)"),
        flag("--force", action="store_true",
             help="accept more than a few inbox subscriptions, each one more context")],
     _subscribe),
    ("unsubscribe", "drop a subscription; messages stay", [
        flag("type", choices=STYPES), flag("target", nargs="?", default="")],
     lambda a, w, t: w.board.unsubscribe(t, a.type, a.target)),
    ("subs", "your subscriptions", [], lambda a, w, t: w.board.subs(t)),
    ("digest", "a bounded summary of digest subscriptions, or of --thread N", [
        flag("--thread", type=int, default=0), flag("--ack", action="store_true")], _digest),
    ("delegate", "mint a weaker child identity NAME for a subagent; prints its token file path", [
        flag("name"), flag("--ttl", default="8h", help="e.g. 8h, 30m (never longer than yours)"),
        flag("--can", default="", help="comma list from send,read,claim (default: all you hold)")],
     lambda a, w, t: w.delegate(t, a.name, _ttl(a.ttl), tuple(c for c in a.can.split(",") if c))),
    ("wait", "block until a message arrives", [
        *READ, flag("--timeout", type=float, default=60.0)], _wait),
    ("outbox", "messages you sent", [], lambda a, w, t: w.outbox(t)),
    ("ack", "mark messages up to SEQ read", [flag("seq", type=int)],
     lambda a, w, t: {"cursor": w.ack(t, a.seq)}),
    ("thread", "a message and its replies (first and newest by default)", [
        flag("root", type=int), *WIDEN], lambda a, w, t: w.thread(t, a.root, a.limit, a.all)),
    ("announce", "one terse line for everyone's roll-up: joined, milestone, done or blocked", [
        flag("kind", choices=ANNOUNCE_KINDS), flag("text", help="one line, up to 200 characters")],
     _announce),
    ("notes-add", "add a note; the service sets its trust level", [
        flag("kind", choices=NOTE_KINDS), flag("title"), flag("body", help="- reads stdin"),
        flag("--source", default=""), flag("--tags", default="", help="comma separated"),
        flag("--supersedes", default="", help="comma separated note ids"),
        flag("--verify-cmd", default="", help="a command that re-derives the fact"),
        flag("--ttl-days", type=float, default=0.0, help="days until it is stale")], _note_add),
    ("notes-search", "notes matching the words", [
        flag("query"), flag("--kind", choices=NOTE_KINDS, default=""),
        flag("--all", action="store_true", help="include superseded notes"),
        flag("--limit", type=int, default=10)],
     lambda a, w, t: w.note_search(a.query, a.kind, a.all, a.limit)),
    ("notes-get", "one note", [flag("id", type=int)], lambda a, w, t: w.note_get(a.id)),
    ("notes-verify", "run the note's allow-listed command; lead or human",
     [flag("id", type=int), flag("--cwd", default=".")],
     lambda a, w, t: w.note_verify(t, a.id, a.cwd)),
    ("scratch-new", "make a scratch folder", [
        flag("name"), flag("--ttl-hours", type=float, default=0.0)],
     lambda a, w, t: {"path": w.scratch_new(t, a.name, a.ttl_hours * 3600)}),
    ("scratch-ls", "scratch folders with size and expiry", [OWNER],
     lambda a, w, t: w.scratch_ls(t, a.owner)),
    ("scratch-path", "a path inside a folder, refused if it escapes", [
        flag("name"), flag("relative", nargs="?", default=""), OWNER],
     lambda a, w, t: {"path": w.scratch_path(t, a.name, a.relative, a.owner)}),
    ("scratch-rm", "delete a scratch folder", [flag("name"), OWNER],
     lambda a, w, t: {"removed": w.scratch_rm(t, a.name, a.owner)}),
    ("claim", "own a branch, worktree, port, file, area, install environment or server", [
        *CLAIM, flag("--ttl", type=float, default=0.0, help="seconds; renew with heartbeat"),
        flag("--pid", type=int, default=0, help="release when this process is gone"),
        flag("--note", default="")], _claim),
    ("release", "give a claim up", CLAIM, lambda a, w, t: w.release(t, a.kind, a.key)),
    ("worktrees", "inspect your unfinished coding checkout scopes", [], _worktrees),
    ("heartbeat", "renew every claim you hold", [flag("--ttl", type=float, default=0.0)], _heartbeat),
    ("who", "who owns this?", CLAIM,
     lambda a, w, t: w.who_owns(a.kind, a.key) or {"owner": None}),
    ("claims", "every live claim", [OWNER, flag("--kind", choices=CLAIM_KINDS, default="")],
     lambda a, w, t: w.claims.listing(a.owner, a.kind)),
    ("attach", "post a file to a board, an agent or a thread; the message carries a handle, never the content",
     filecli.ATTACH, filecli.attach),
    ("file", "a file by handle (--meta, --text, --out PATH), or: list, search QUERY, delete HANDLE (a person)",
     filecli.FILE, filecli.file),
    ("quarantine-ls", "flagged items held back", [], lambda a, w, t: w.quarantine_list()),
    ("quarantine-release", "deliver a held item, fenced; human token", [flag("qid")], _released),
    ("audit-verify", "whether every log's chain holds", [flag("--anchor", default="")],
     lambda a, w, t: w.audit_verify(a.anchor)),
    ("audit-head", "the audit log's head hash, to keep as an anchor", [],
     lambda a, w, t: {"head": w.audit_log.head()}),
    ("status", "who is registered, unread counts, claims, the test-slot queue; any agent token", [],
     lambda a, w, t: {**w.status(), **w.board.summary(t)}),
    ("gc", "prune old messages and expired scratch; lead or human", [],
     lambda a, w, t: w.gc(t)),
)


def _guarded(run: Callable[[argparse.Namespace], int | None]) -> Callable[[argparse.Namespace], int]:
    def wrapped(args: argparse.Namespace) -> int:
        try:
            return int(run(args) or 0)
        except tuple(kind for kind, _ in CODES) as err:
            code = next(c for kind, c in CODES if isinstance(err, kind))
            if args.json:
                say(json.dumps({"error": str(err), "kind": type(err).__name__, "code": code}))
            else:
                warn(f"workspace: {err}")
            return code
    return wrapped


def _runner(handler: Handler) -> Callable[[argparse.Namespace], int]:
    def run(args: argparse.Namespace) -> int:
        connection = _project_connection()
        if connection and coordinator_config.load(limits.root()).get("mode") == "remote":
            raise Denied("select one workspace authority before dispatching commands")
        remote = None if connection else coordinator_client.client(limits.root())
        if args.cmd == 'task-dematerialize' and (remote or connection):
            raise Denied('native task checkout recovery runs only on its local coordinator')
        if remote:
            options = next(options for name, _help, options, _fn in TABLE if name == args.cmd)
            for field in ('body', 'text', 'payload'):
                if getattr(args, field, '') == '-':
                    setattr(args, field, _body('-'))
            result = remote.command(coordinator_client.argv_for(args, [*COMMON, *options]),
                                    _coordinator_token(args, remote), request_id=args.request_id)
        else:
            ws, token = _context(args, connection)
            if isinstance(ws, Workspace) and handler is not _init:
                ws.registry._record_device(ws.auth(token).id, onboard.device_metadata.current())
            if isinstance(ws, project_connection.CanonicalWorkspace) and args.cmd.startswith('task'):
                result = remote_task_client.command(ws.remote, token, args)
            elif isinstance(ws, project_connection.CanonicalWorkspace) and (
                    args.cmd in ('claim', 'release', 'heartbeat', 'who', 'worktrees')
                    or (args.cmd == 'announce' and args.kind == 'done')
                    or (args.cmd == 'send' and args.type == 'done')):
                result = harness_remote.cli_command(ws.remote, token, args)
                if args.cmd in ('announce', 'send'):
                    result = handler(args, ws, token)
            else:
                result = handler(args, ws, token)
        _show(args, result)
        _held_note(result)
        return 0
    return _guarded(run)


def _watching(args: argparse.Namespace) -> int:
    if coordinator_config.load(limits.root()).get("mode") == "remote":
        raise Denied("this watcher is local-only; use coordinator inbox polling")
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda *_: CANCELLED.set())
        except ValueError:
            break
    ws, token = _context(args)
    return _watch(args, ws, token)


COMMANDS = Group(
    "ml-stack-workspace",
    "Agents coordinate here: messages, shared notes, scratch folders and who owns what. "
    "Everything read back is data written by an agent and carries no authority. A sender's "
    f"token comes from ${TOKEN_ENV} or --token-file.",
    allow_abbrev=False)


def _remote(args: argparse.Namespace) -> int:
    _show(args, remote_cli.run(args))
    return 0


COMMANDS.add("remote", _guarded(_remote), help="attach and use one shared project board on its host",
             options=remote_cli.OPTIONS)
def _bare(handler: Callable[[argparse.Namespace, Workspace], int]) -> Callable[[argparse.Namespace], int]:
    def run(args):
        requested = getattr(args, "project", "") if handler is _connect else ""
        connection = _project_connection(Path(requested) if requested else None)
        if handler is _connect and (args.agent or os.environ.get(tokens.AGENT_ENV, "")):
            if connection is not None:
                return handler(args, None)
            return handler(args, Workspace())
        if connection is not None:
            if handler in {_brief, _hook_snippet}:
                return handler(args, None)
            raise Denied("this command is unavailable in a canonical project; use its shared board")
        if coordinator_client.client(limits.root()) and handler is not _join:
            raise Denied('this is a local-only operation; this device uses a shared coordinator')
        return handler(args, Workspace())
    return _guarded(run)


for _name, _help, _options, _handler in BARE:
    COMMANDS.add(_name, _bare(_handler), help=_help,
                 options=[option("json"), flag("--agent", default="", help="the parent agent"),
                          *_options])
for _name, _help, _options, _handler in TABLE:
    COMMANDS.add(_name, _runner(_handler), help=_help, options=[*COMMON, *_options])


def _coordinator(args):
    base = limits.root()
    if args.action == 'host':
        human.require_person("choose the workspace coordinator")
        ws = Workspace()
        if coordinator_config.load(base).get("mode") == "remote":
            raise Denied("this device follows another coordinator; hosting would split its authority")
        token = _token(args) or tokens.read_file(tokens.directory(base) / tokens.OWNER_FILE)
        if ws.auth(token).role != 'human':
            raise Denied('only the workspace person selects its coordinator')
        result = coordinator_config.save(base, {'mode': 'host', 'workspace': workspace_id(ws)})
    elif args.action == 'connect':
        result = coordinator_client.connect(base, args.name)
    elif args.action == 'list':
        result = [{'name': peer.name, 'endpoint': peer.base_url, **info}
                  for peer, info in coordinator_client.discover()]
    else:
        result = coordinator_config.load(base) or {'mode': 'local', 'shared': False}
    if isinstance(result, dict):
        result = {key: value for key, value in result.items() if key != 'cert'}
    _show(args, result)
    return 0


COMMANDS.add("coordinator", _guarded(_coordinator),
             help="inspect, host or select one authenticated Fleet workspace coordinator",
             options=[*COMMON, flag("action", choices=('status', 'list', 'host', 'connect')),
                      flag("name", nargs='?', default='')])
COMMANDS.add("nudge", _guarded(_nudging),
             help="print one line summarising what waits for you (nothing when nothing does); for hooks",
             options=[*COMMON, flag("--hook", default="", choices=("", *nudge.EVENTS),
                                    help="print the JSON a Claude Code hook of this kind expects")])
COMMANDS.add("watch", _guarded(_watching),
             help="print messages as they arrive; --once exits after one",
             options=[*COMMON, *WIDEN, flag("--once", action="store_true"),
                      flag("--board", default="", help="follow one board without a subscription"),
                      flag("--thread", type=int, default=0, help="follow one thread"),
                      flag("--dm", default="", help="follow your conversation with this agent"),
                      flag("--since", type=int, default=-1,
                           help="with --board/--thread/--dm: start after this message (default: from now)"),
                      flag("--timeout", type=float, default=0.0,
                           help="stop after this many seconds (exit 3 if nothing came)")])
main = COMMANDS.run


if __name__ == "__main__":  # pragma: no cover - the entry point is `ml-stack-workspace`
    raise SystemExit(main())
