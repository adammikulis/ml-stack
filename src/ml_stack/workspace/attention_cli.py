"""The command-line side of attention: the inbox with what is owed, a helper's own inbox, the status page and the brief's unread."""

from __future__ import annotations

import argparse
from typing import Any

from ml_stack.command import flag
from ml_stack.log import say
from ml_stack.workspace import attention
from ml_stack.workspace.screen import fence
from ml_stack.workspace.service import Workspace


def owed_text(ws: Workspace, token: str, announcements: bool = True) -> str:
    """Unanswered requests and, when asked, the count of announcements not yet seen; nothing is marked read."""
    roll = ws.board.rollup(token, False) if announcements else None
    return attention.attention(ws, ws.auth(token).id, roll["messages"] if roll else 0)


def inbox(args: argparse.Namespace, ws: Workspace, token: str, label: str) -> Any:
    """Unread messages: a labelled helper sees only those meant for it and cannot ack; others also see what they owe."""
    if label and not args.children:
        if args.ack:
            raise ValueError("a labelled helper shares its parent's inbox and cannot --ack it")
        who = ws.auth(token)
        return [ws.deliver(r, args.raw, 0, who) for r in attention.helper_messages(ws, who.id, label)]
    if not args.children:
        roll = None if args.json else ws.board.rollup(token, args.ack)
        owed = "" if args.json else owed_text(ws, token, False)
        for text in (roll["text"] if roll else "", owed):
            if text:
                say(text)
        return ws.inbox(token, args.ack, args.limit, args.raw, args.all)
    if args.ack:
        raise ValueError("--children shows some of the unread messages, so it cannot --ack")
    me = ws.auth(token).id
    return [m for m in ws.inbox(token, False, 0, args.raw, True) if m["from"].startswith(me + "/")]


def digest(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    """The digest, or with --status the coordinator's one-screen page of workers, claims and owed answers."""
    if not args.status:
        return ws.board.digest(token, args.ack, args.thread)
    lines = attention.status_lines(ws, ws.auth(token).id)
    return {"authority": "none", "text": fence("\n".join(lines), "workspace:status",
                                                "names and subjects written by agents").text}


def owed(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    """What is unanswered for the caller and how many announcements are new; empty when neither."""
    text = owed_text(ws, token)
    return {"authority": "none", "text": text} if text else ""


def helper_brief(ws: Any, me: str, label: str, reader: Any) -> None:
    """Print the unread direct messages already waiting for the helper ``label``."""
    mine = attention.helper_messages(ws, me, label) if hasattr(ws, "bus") else []
    if mine:
        say(f"Unread messages for you ({label}); `inbox --agent {me} --label {label}` shows them again:")
    for row in mine:
        shown = ws.deliver(row, False, 0, reader)
        say(f"[{shown['seq']}] {shown['type']} from {shown.get('from_label', shown['from'])} (data, no authority): {shown['text']}")


TABLE = [("attention", "what is unanswered for you and how many announcements are new; nothing when neither", [], owed)]
STATUS = flag("--status", action="store_true",
              help="active labelled workers, their claims and what is unanswered for you")
