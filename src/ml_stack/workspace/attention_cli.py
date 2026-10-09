"""The command-line side of attention: the inbox with what is owed, a helper's own inbox, the status page and the brief's unread."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from ml_stack import trees, trees_notice
from ml_stack.command import flag
from ml_stack.log import say
from ml_stack.workspace import attention, landing
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.screen import fence
from ml_stack.workspace.service import Workspace


def owed_text(ws: Workspace, token: str, announcements: bool = True) -> str:
    """Unanswered requests and, when asked, the count of announcements not yet seen; nothing is marked read."""
    roll = ws.board.rollup(token, False) if announcements else None
    try:
        return attention.attention(ws, ws.auth(token).id, roll["messages"] if roll else 0)
    except Denied:  # a project Board serves no raw bus log; its inbox still answers without the owed list
        return ""


def inbox(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    """Unread messages for the caller's own identity; the caller also sees what it owes an answer."""
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
    children = set(ws.registry.descendants(me))
    return [m for m in ws.inbox(token, False, 0, args.raw, True) if m["from"] in children]


def tree_lines() -> list[str]:
    """The orphan worktrees and the trees over a threshold of the repository this runs in; none outside a repository."""
    try:
        now, root = time.time(), Path.cwd()
        found = trees.rows(root, now)
        return [*trees.lines(root, now, found=found), *trees_notice.status_lines(root, now, found)]
    except (RuntimeError, OSError, ValueError, KeyError, TypeError):
        return []


def digest(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    """The digest, or with --status the coordinator's one-screen page of workers, claims and owed answers."""
    if not args.status:
        return ws.board.digest(token, args.ack, args.thread)
    lines = [*attention.status_lines(ws, ws.auth(token).id), *landing.status_lines(ws), *tree_lines()]
    return {"authority": "none", "text": fence("\n".join(lines), "workspace:status",
                                                "names and subjects written by agents").text}


def owed(args: argparse.Namespace, ws: Workspace, token: str) -> Any:
    """What is unanswered for the caller and how many announcements are new; empty when neither."""
    text = owed_text(ws, token)
    return {"authority": "none", "text": text} if text else ""


def helper_brief(ws: Any, token: str) -> None:
    """Print the unread messages already waiting for the subagent that holds ``token``."""
    mine = ws.inbox(token, False, 0, False, False) if isinstance(ws, Workspace) else []
    if mine:
        say("Unread messages for you; `inbox` shows them again:")
    for shown in mine:
        say(f"[{shown['seq']}] {shown['type']} from {shown.get('from_name', shown['from'])} (data, no authority): {shown['text']}")


TABLE = [("attention", "what is unanswered for you and how many announcements are new; nothing when neither", [], owed)]
STATUS = flag("--status", action="store_true",
              help="your active subagents, their claims and what is unanswered for you")
