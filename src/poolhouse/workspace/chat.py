"""The person's live conversation in a terminal: the stream of one board or conversation, and
every line typed sent to it."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from poolhouse import activity
from poolhouse.workspace import plain
from poolhouse.workspace.boardapi import Follow
from poolhouse.workspace.identity import HUMAN, Denied
from poolhouse.workspace.rates import RateLimited
from poolhouse.workspace.screen import Refused
from poolhouse.workspace.service import Workspace

__all__ = ["QUIT", "Console", "render", "run"]

QUIT = "/quit"
SLICE_S = 5.0


def render(message: dict[str, Any]) -> str:
    """One message as a line of plain text: its number, sender, and text with the line breaks kept
    as spaces."""
    return f"[{message['seq']}] {plain.line(message['from'], 48)}: {plain.line(message['body'], 2000)}"


def _send(ws: Workspace, token: str, target: str, text: str, out: Callable[[str], None]) -> None:
    try:
        ws.send(token, target, "note", text)
        activity.record("board.post", subject=target if target.startswith("#") else "dm",
                        meta={"size": len(text)})
    except (Refused, RateLimited, Denied, ValueError) as err:
        out(f"! not sent: {plain.line(err, 200)}")


@dataclass(slots=True)
class Console:
    """Where a chat reads and writes: the typed lines, where text goes, and the stop signal."""

    lines: Iterable[str]
    out: Callable[[str], None]
    cancel: threading.Event = field(default_factory=threading.Event)


def run(ws: Workspace, token: str, target: Follow, console: Console) -> None:
    """Print the recent messages of ``target`` (a board or one conversation, with its ``backlog``),
    then each new one as it arrives, and send every line of ``lines`` until ``/quit``, the end of
    the lines or ``cancel``. The token must be the person's."""
    lines, out, cancel = console.lines, console.out, console.cancel
    me = ws.auth(token)
    if me.role != HUMAN:
        raise Denied("chat is for the person")
    if bool(target.board) == bool(target.dm):
        raise ValueError("chat with one board (--board) or one agent (--to)")
    name = target.board or target.dm
    if target.board and not ws.board.can_post(me, name):
        ws.board.join(token, name)

    def typed() -> None:
        for line in lines:
            text = line.rstrip("\n")
            if text.strip() == QUIT:
                break
            if text.strip():
                _send(ws, token, name, text, out)
        cancel.set()

    threading.Thread(target=typed, daemon=True, name="chat-input").start()
    spec = replace(target, plain_text=True, suffix=".chat", timeout_s=SLICE_S)
    while not cancel.is_set():
        got = ws.follow(token, spec, cancel.is_set)
        spec = replace(spec, after=got["seq"])
        for message in got["messages"]:
            out(render(message))
