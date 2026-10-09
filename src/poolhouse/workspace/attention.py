"""What a coordinating agent owes the board: unanswered direct requests and its live subagents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from poolhouse.workspace.boardapi import data_line

if TYPE_CHECKING:
    from poolhouse.board.session import Session

__all__ = ["ACTIVE_S", "age", "owed_lines", "status_lines", "workers"]

ACTIVE_S = 12 * 3600.0
SHOWN = 8
ANNOUNCEMENTS = "#announcements"


def age(seconds: float) -> str:
    """A compact age such as 14m, 3h or 2d."""
    return f"{int(seconds // 86400)}d" if seconds >= 86400 else f"{int(seconds // 3600)}h" if seconds >= 3600 \
        else f"{int(seconds // 60)}m"


def owed_lines(owed: list[dict[str, Any]]) -> list[str]:
    """One line per unanswered request, the oldest first, at most SHOWN of them."""
    lines = [f"[{o['seq']}] {o['type']} from {o['from']}, unanswered {age(o['age_s'])}: {o['subject']}"
             for o in sorted(owed, key=lambda o: -o["age_s"])[:SHOWN]]
    return [*lines, f"(+{len(owed) - SHOWN} more)"] if len(owed) > SHOWN else lines


def workers(s: Session) -> dict[str, Any]:
    """The newest announcement of every live subagent below this session not yet done and heard from within ACTIVE_S."""
    now = s.clock()
    known = {a.name: a for a in s.agents()}
    below: set[str] = set()
    grew = True
    while grew:
        grew = False
        for a in known.values():
            if a.name not in below and (a.parent == s.name or a.parent in below):
                below.add(a.name)
                grew = True
    last: dict[str, Any] = {}
    for e in s.read(channel=ANNOUNCEMENTS).entries:
        if e.sender in below and e.at_ms / 1000 > now - ACTIVE_S:
            last[e.sender] = e
    return {name: e for name, e in last.items() if e.fields.get("type") != "done"}


def status_lines(s: Session, owed: list[dict[str, Any]]) -> list[str]:
    """One screen: active subagents, their last announcement age and claims, then what is unanswered for the session."""
    now = s.clock()
    held = s.claims()
    lines = []
    for name, e in sorted(workers(s).items(), key=lambda kv: -kv[1].at_ms):
        mine = [f"{c.kind}:{c.key}" for c in held if c.owner == name]
        lines.append(f"{name}: {e.fields.get('type')} {age(now - e.at_ms / 1000)} ago"
                     f"{', claims ' + ', '.join(mine) if mine else ''}: {data_line(e.text, 80)}")
    return [f"Active subagents ({len(lines)}):", *(lines or ["(none)"]),
            f"Unanswered for {s.name} ({len(owed)}):", *(owed_lines(owed) or ["(none)"])]
