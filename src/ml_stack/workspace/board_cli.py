"""The `ml-stack-workspace` commands that run on the node: who you are, messages, agents, spawn, retire."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import trees, trees_notice
from ml_stack.board import credentials, session
from ml_stack.board.client import Conflict, NodeError
from ml_stack.board.session import Agent, Entry, Native, Session
from ml_stack.command import flag
from ml_stack.log import say, warn
from ml_stack.workspace import (
    agent_display,
    agent_filter,
    attention,
    limits,
    nudge,
    onboard,
    render,
    session_name,
)
from ml_stack.workspace.boardapi import Held, data_line
from ml_stack.workspace.cli_options import FOR_AGENT, READ
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.modelid import clean_harness, clean_model, describe
from ml_stack.workspace.screen import Refused, fence, refusals

__all__ = ["TABLE", "guarded", "runner"]

TYPES = ("task", "status", "handoff", "question", "answer", "claim", "release", "note", "file")
ANNOUNCE_KINDS = ("joined", "milestone", "done", "blocked")
ANNOUNCE = "#announcements"
Handler = Callable[[argparse.Namespace, Session], Any]


def exit_code(err: Exception) -> int:
    """The exit status for a refusal: 2 malformed, 3 not allowed, 4 over a quota, 5 held by another."""
    if isinstance(err, Conflict):
        return 5
    if isinstance(err, NodeError):
        return {"invalid": 2, "quota": 4}.get(err.code, 3)
    if isinstance(err, (Refused, Denied)):
        return 3
    return 2 if isinstance(err, (ValueError, EOFError)) else 3


FAILURES = (NodeError, OSError, Denied, Refused, ValueError, EOFError)


def guarded(run: Callable[[argparse.Namespace], int | None]) -> Callable[[argparse.Namespace], int]:
    """``run`` with its refusals turned into one warning (or one JSON object) and an exit status."""
    def wrapped(args: argparse.Namespace) -> int:
        try:
            session_name.check_agent(getattr(args, "agent", ""))
            return int(run(args) or 0)
        except FAILURES as err:
            code = exit_code(err)
            if getattr(args, "json", False):
                say(json.dumps({"error": str(err), "kind": type(err).__name__, "code": code}))
            else:
                warn(f"workspace: {err}")
            return code
    return wrapped


def connect(args: argparse.Namespace) -> Session:
    """The caller's session on the board its directory belongs to."""
    return session.connect(token_file=args.token_file, agent=args.agent, board=args.board)


def runner(handler: Handler) -> Callable[[argparse.Namespace], int]:
    """A command that connects, runs ``handler`` and prints what it returns."""
    def run(args: argparse.Namespace) -> int:
        s = connect(args)
        result = handler(args, s)
        if result is not None:
            named = getattr(args, "for_agent", "")
            if named:
                result = agent_filter.narrow(result, agent_filter.resolve([a.name for a in s.agents(True)], named))
            render.show(args, result)
            render.held_note(result)
        return 0
    return guarded(run)


def screened(what: str, *texts: str) -> None:
    """Refuse text that is too long, holds a credential or names a private term."""
    lim = limits.load(limits.root())
    joined = "\n".join(texts)
    if len(joined.encode()) > lim.body_bytes:
        raise Refused(f"{what} is {len(joined.encode())} bytes; the limit is {lim.body_bytes}")
    why = refusals(joined, limits.denylist_path(limits.root()))
    if why:
        raise Refused(f"{what} was not written: {'; '.join(why)}. Remove it and send again.")


def shown(e: Entry, models: dict[str, Agent], cap: int = 0) -> dict[str, Any]:
    """A message as a reader gets it: fenced as untrusted data, never as an instruction."""
    kind, subject, body = str(e.fields.get("type", "note")), str(e.fields.get("subject", "")), e.text
    if cap and len(body) > cap:
        body = f"{body[:cap]}…({len(body) - cap} more chars; {e.ref})"
    who = models.get(e.sender)
    model, state = (who.model, who.model_state) if who else ("", "")
    body = f"subject: {subject}\n{body}" if subject else body
    screened_text = fence(body, f"workspace:{e.sender}#{e.ref}", f"agent {e.sender} ({describe(model, state)}), {kind}")
    return {"seq": e.ref, "type": kind, "from": e.sender, "from_name": e.sender, "from_role": "agent", "from_model": model,
            "from_model_state": state, "to": e.fields.get("to", ""), "ts": e.at_ms / 1000, "reply_to": e.fields.get("reply_id", ""),
            "trust": "agent-claimed", "authority": "none", "state": "clear", "flags": list(screened_text.markers),
            "text": screened_text.text}


def names(s: Session) -> dict[str, Agent]:
    return {a.name: a for a in s.agents(True)}


def rollup(s: Session, ack: bool) -> str:
    """The newest few unseen announcements as one fenced block, or an empty string."""
    page = s.read(s.cursor("announce"), channel=ANNOUNCE)
    fresh = [e for e in page.entries if e.sender != s.name]
    if ack and page.entries:
        s.keep("announce", page.cursor)
    if not fresh:
        return ""
    top = limits.load(limits.root()).announce_rollup
    lines = [f"[{e.ref}] {e.fields.get('type', '')} {e.sender}: {data_line(e.text, 200)}" for e in fresh[-top:]]
    if len(fresh) > top:
        lines.insert(0, f"(+{len(fresh) - top} older announcements; `digest` lists them)")
    return render.block(lines, "announcements")


def unanswered(s: Session) -> list[dict[str, Any]]:
    """Direct questions, tasks and handoffs to this session older than the owner's `owed_after_s` that it never
    answered (it wrote to the sender, or in reply, after the request)."""
    now, older_s = s.clock(), limits.load(limits.root()).owed_after_s
    sent = s.read(channel="#dm", by=s.name).entries
    owed = []
    for e in s.read(inbox=True).entries:
        age = now - e.at_ms / 1000
        if e.fields.get("type") not in nudge.URGENT or age < older_s:
            continue
        if not any(m.at_ms > e.at_ms and (m.fields.get("to") == e.sender or m.fields.get("reply_id") == e.ref) for m in sent):
            owed.append({"seq": e.ref, "from": e.sender, "type": e.fields.get("type"), "age_s": age,
                         "subject": data_line(e.fields.get("subject") or e.text, 60)})
    return owed


def whoami(args: argparse.Namespace, s: Session) -> Any:
    """Who the token says you are; --model and --harness record what you say you run on."""
    a = s.whoami(clean_model(args.model) if args.model else "", clean_harness(args.harness))
    return {"id": a.name, "parent": a.parent, "board": s.board, "model": a.model or "unknown", "model_state": a.model_state,
            "harness": a.harness}


def agents(args: argparse.Namespace, s: Session) -> Any:
    """Every live identity with its model and whether it may coordinate."""
    return [{"id": a.name, "role": "agent", "parent": a.parent, "model": a.model, "model_state": a.model_state,
             "harness": a.harness, "last_acted": a.seen_ms / 1000, "retired": a.retired, **agent_display.describe(a)}
            for a in s.agents()]


def spawn(args: argparse.Namespace, s: Session) -> Any:
    """Register a subagent's native session as a child of the caller."""
    made = s.register(Native(args.model, args.harness, args.session))
    return {"id": made.name, "parent": made.parent}


def retire(args: argparse.Namespace, s: Session) -> Any:
    """End the subagent the caller runs as."""
    name = s.name
    done = s.retire()
    credentials.forget(s.client.state, s.board, name)
    return {"id": done["name"], "tokens_revoked": done["tokens_revoked"], "leases_released": done["leases_released"]}


def announce(args: argparse.Namespace, s: Session) -> Any:
    """Post one terse line for everyone's roll-up."""
    text = render.body(args.text)
    chars = limits.load(limits.root()).announce_chars
    if args.kind not in ANNOUNCE_KINDS:
        raise Refused(f"an announcement is one of {', '.join(ANNOUNCE_KINDS)}, not {args.kind!r}; "
                      f"for anything else send a direct message to the one agent who needs it")
    if not text.strip() or "\n" in text or len(text) > chars:
        raise Refused(f"an announcement is one line of at most {chars} characters ({len(text)} given); put the "
                      f"detail in a note and link it, such as 'done: see note 12'")
    screened("the announcement", text)
    sent = s.post(ANNOUNCE, args.kind, text, subject=args.kind)
    return {"seq": sent["id"].rsplit(":", 1)[-1], "type": args.kind, "to": ANNOUNCE, "from": sent["sender"]}


def send(args: argparse.Namespace, s: Session) -> Any:
    """Send a message to one agent, or an announcement to `*`."""
    if args.to == "*":
        return announce(argparse.Namespace(kind=args.type, text=args.body), s)
    if args.type not in TYPES:
        raise ValueError(f"type must be one of {', '.join(TYPES)}")
    if args.type == "file":
        raise Refused("a file message is made by `attach`")
    body = render.body(args.body)
    screened("the message", args.subject, body)
    sent = s.post(args.to, args.type, body, subject=args.subject, reply_id=args.reply_to)
    return {"seq": sent["id"].rsplit(":", 1)[-1], "type": args.type, "to": args.to, "from": sent["sender"]}


def inbox(args: argparse.Namespace, s: Session) -> Any:
    """Unread direct messages, fenced as data; --ack marks what is shown read."""
    if args.children and args.ack:
        raise ValueError("--children shows some of the unread messages, so it cannot --ack")
    lim = limits.load(limits.root())
    known = names(s)
    page = s.read(s.cursor("inbox"), inbox=True)
    entries = page.entries
    if args.children:
        mine = {a.name for a in known.values() if a.parent == s.name}
        mine |= {a.name for a in known.values() if a.parent in mine}
        entries = [e for e in entries if e.sender in mine]
    wide = args.all
    take = args.limit or (len(entries) if wide else lim.read_items)
    out, used = Held(), 0
    for e in entries[:take]:
        item = shown(e, known, 0 if wide else lim.read_item_chars)
        if not wide and out and used + len(item["text"]) > lim.read_total_chars:
            break
        used += len(item["text"])
        out.append(item)
    out.held = len(entries) - len(out)
    if not args.json:
        for text in (rollup(s, args.ack), owed_text(s)):
            if text:
                say(text)
    if args.ack and out:
        s.keep("inbox", advanced(page.cursor, entries[:len(out)], s.cursor("inbox")))
    return out


def advanced(cursor: dict[str, int], read: list[Entry], before: dict[str, int]) -> dict[str, int]:
    """``before`` moved forward over the entries ``read`` (not past ones that were held back)."""
    if len(read) == 0:
        return before
    moved = dict(before)
    for e in read:
        origin, seq = e.id.rsplit(":", 2)[-2:]
        moved[origin] = max(moved.get(origin, 0), int(seq))
    return moved


def owed_text(s: Session, announcements: int = 0) -> str:
    """Unanswered requests as one fenced block, or an empty string."""
    owed = unanswered(s)
    if not owed:
        return ""
    lines = [f"Unanswered for you ({len(owed)}); answer with `ml-stack-workspace send AGENT answer TEXT --reply-to SEQ`:",
             *attention.owed_lines(owed)]
    return fence("\n".join(lines), "workspace:attention", "requests written by agents").text


def attend(args: argparse.Namespace, s: Session) -> Any:
    """What is unanswered for you and how many announcements are new; nothing when neither."""
    text = owed_text(s)
    new = len([e for e in s.read(s.cursor("announce"), channel=ANNOUNCE).entries if e.sender != s.name])
    if new:
        text = (text + "\n" if text else "") + f"{new} new announcements (`ml-stack-workspace inbox` shows them)."
    return {"authority": "none", "text": text} if text else ""


def tree_lines() -> list[str]:
    """The orphan worktrees and the trees over a threshold of the repository this runs in; none outside a repository."""
    try:
        now, root = time.time(), Path.cwd()
        found = trees.rows(root, now)
        return [*trees.lines(root, now, found=found), *trees_notice.status_lines(root, now, found)]
    except (RuntimeError, OSError, ValueError, KeyError, TypeError):
        return []


def digest(args: argparse.Namespace, s: Session) -> Any:
    """What is new: the announcements; with --status, your active subagents, their claims and what you owe."""
    if not args.status:
        text = rollup(s, args.ack)
        return {"authority": "none", "text": text or "nothing new"}
    return {"authority": "none", "text": render.block([*attention.status_lines(s, owed=unanswered(s)), *tree_lines()], "status")}


def waiting(s: Session) -> dict[str, Any]:
    """What waits unread, as `nudge.Waiting` takes it; nothing is marked read."""
    entries = s.read(s.cursor("inbox"), inbox=True).entries
    known = names(s)
    rows = [{"seq": int(e.ref) if e.ref.isdigit() else 0, "to": e.fields.get("to"), "from": e.sender,
             "type": e.fields.get("type"), "ts": e.at_ms / 1000} for e in entries]
    messages = [{"seq": r["seq"], "type": r["type"], "from": r["from"], "from_name": r["from"],
                 "text": f"{e.fields.get('subject')}: {e.text[:600]}" if e.fields.get("subject") else e.text[:600]}
                for r, e in zip(rows, entries, strict=True) if not shown(e, known)["flags"]][:50]
    return {"me": s.name, "now": s.clock(), "rows": rows, "messages": messages}


def brief(args: argparse.Namespace, s: Session) -> Any:
    """The brief of the spawned subagent the caller runs as, then what already waits for it."""
    me = next((a for a in s.agents() if a.name == s.name), None)
    if me is None or not me.parent:
        raise ValueError(f"{s.name} was not spawned by another agent; run `spawn` as the parent, then brief --agent NAME")
    say(onboard.brief(me.name, me.parent, args.registered), end="")
    mine = s.read(s.cursor("inbox"), inbox=True).entries
    if mine:
        say("Unread messages for you; `inbox` shows them again:")
    known = names(s)
    for e in mine:
        say(f"[{e.ref}] {e.fields.get('type')} from {e.sender} (data, no authority): {shown(e, known)['text']}")


def nudging(args: argparse.Namespace) -> int:
    """One line summarising what waits (nothing when nothing does); with --hook, the JSON a harness hook expects."""
    stdin = sys.stdin.read() if args.hook == "stop" and not sys.stdin.isatty() else ""
    try:
        summary = waiting(connect(args))
    except FAILURES:
        if args.hook:
            return 0
        raise
    if args.hook:
        out = nudge.output(args.hook, nudge.Waiting.of(summary), stdin)
    else:
        out = nudge.Waiting.of(summary).line()
    if out:
        say(out)
    return 0


TABLE: tuple[tuple[str, str, list[Any], Handler], ...] = (
    ("whoami", "who the token says you are; --model records your own model id as claimed",
     [flag("--model", default="", help="your own model id, recorded as claimed"),
      flag("--harness", default="", help="your harness, e.g. claude-code or codex")], whoami),
    ("spawn", "register a subagent's native session as your child: the board names it and records you as its parent; "
              "run by a harness hook for the session that started the subagent",
     [flag("--session", help="the subagent's native session id (its agent id)"),
      flag("--harness", default="claude-code", help="the harness the subagent runs in"),
      flag("--model", default="", help="the subagent's model id, recorded as claimed")], spawn),
    ("retire", "end the subagent you run as: its token stops working and its claims are released; run by its stop hook",
     [], retire),
    ("agents", "every live identity with its model and whether the model is verified; --for-agent shows one",
     [FOR_AGENT], agents),
    ("send", "send a message (BODY - reads stdin)", [
        flag("to", help="an agent id, or * for the announcements board (joined, milestone, done, blocked only)"),
        flag("type", choices=(*TYPES, *ANNOUNCE_KINDS)), flag("body"), flag("--subject", default=""),
        flag("--reply-to", default="", help="the number of the message this answers")], send),
    ("announce", "one terse line for everyone's roll-up: joined, milestone, done or blocked", [
        flag("kind", choices=ANNOUNCE_KINDS), flag("text", help="one line, up to 200 characters")], announce),
    ("inbox", "unread direct messages, fenced as data", [
        *READ, FOR_AGENT, flag("--children", action="store_true", help="only messages from your delegates")], inbox),
    ("attention", "what is unanswered for you and how many announcements are new; nothing when neither", [], attend),
    ("digest", "what is new since you last looked; --status for your subagents, their claims and what you owe", [
        FOR_AGENT, flag("--ack", action="store_true"),
        flag("--status", action="store_true", help="your active subagents, their claims and what is unanswered for you")],
     digest),
    ("brief", "print the short brief for the spawned subagent you run as (--agent NAME), to paste into its prompt",
     [flag("--registered", action="store_true", help="hooks registered the subagent and record its joined and done")],
     brief),
)
