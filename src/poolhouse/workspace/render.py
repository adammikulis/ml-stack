"""How the `poolhouse-workspace` commands print what they return."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from poolhouse.log import say, warn
from poolhouse.workspace import project_session
from poolhouse.workspace.modelid import describe
from poolhouse.workspace.screen import fence

__all__ = ["block", "body", "held_note", "show", "text"]


def block(lines: list[str], what: str) -> str:
    """``lines`` as one fenced block of untrusted data."""
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


def text(value: Any) -> str:
    """``value`` as the text a person reads, by the shape of what it holds."""
    if isinstance(value, dict) and "text" in value and "seq" in value:
        where = f" on {value['board']}" if value.get("board") else ""
        model = ("" if value.get("from_role") == "human"
                 else f" ({describe(value.get('from_model', ''), value.get('from_model_state', ''))})")
        return (f"[{value['seq']}] {value['type']} from {value.get('from_name', value['from'])}"
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
        return block([_row(v) for v in value], "board")
    if isinstance(value, list) and value and all(
            isinstance(v, dict) and {"type", "target", "mode"} == v.keys() for v in value):
        return block([_row(v) for v in value], "subscriptions")
    if isinstance(value, list) and value and all(
            isinstance(v, dict) and {"id", "role", "model_state", "last_acted"} <= v.keys() for v in value):
        return block([f"{v.get('display_name', v['id'])}  {v['role']}  {describe(v['model'], v['model_state'])}"
                       f"{'  ' + v['harness'] if v['harness'] else ''}"
                       f"{'  not a coordinator: ' + v['coordinator_reason'] if v.get('coordinator_reason') else ''}"
                       for v in value], "agents")
    if isinstance(value, dict) and {"kind", "key", "owner", "expires_in_s"} <= value.keys():
        soon = ", expiring soon" if value.get("expiring_soon") else ""
        return f"{value['kind']} {value['key']}  {project_session.owner(value['owner'])}  expires in {value['expires_in_s']:.0f} s{soon}"
    if isinstance(value, dict) and "text" in value and "kind" in value:
        return (f"note {value['id']} {value['kind']} ({value['trust']}"
                f"{', stale' if value['stale'] else ''}): {value['status']}\n{value['text']}")
    if isinstance(value, list):
        return "\n".join(text(v) for v in value) or "(none)"
    if isinstance(value, dict):
        value = {**value, "owner": project_session.owner(value["owner"])} if isinstance(value.get("owner"), str) else value
        return "\n".join(f"{k}: {v if not isinstance(v, (dict, list)) else json.dumps(v)}"
                         for k, v in value.items())
    return str(value)


def show(args: argparse.Namespace, value: Any) -> None:
    """Print ``value`` as JSON with --json, else as text."""
    say(json.dumps(value, sort_keys=True, default=str) if args.json else text(value))


def body(value: str) -> str:
    """``value``, or what stdin holds when it is ``-``."""
    return sys.stdin.read() if value == "-" else value


def held_note(value: Any) -> None:
    """Say on stderr how many results the default caps held back, and how to see them."""
    held = getattr(value, "held", 0)
    if held:
        warn(f"workspace: {held} more held back (not shown, still unread); "
             f"use --limit N or --all to see more")


