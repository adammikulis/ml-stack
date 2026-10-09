"""What a coordinating agent owes the board: unanswered direct requests and its live subagents."""

from __future__ import annotations

from typing import Any

from ml_stack.workspace.boardapi import data_line
from ml_stack.workspace.boards import ANNOUNCE
from ml_stack.workspace.nudge import URGENT
from ml_stack.workspace.screen import fence

OWED_AFTER_S = 600.0
LOOKBACK_S = 7 * 86400.0
ACTIVE_S = 12 * 3600.0
SHOWN = 8


def _mine(row: dict[str, Any], me: str) -> bool:
    return row["kind"] == "msg" and row["from"] == me


def _answered(row: dict[str, Any], later: list[dict[str, Any]], me: str) -> bool:
    root = int(row.get("thread") or row["seq"])
    return any(_mine(r, me) and (r["to"] == row["from"] or int(r.get("reply_to") or 0) == row["seq"]
                                 or int(r.get("thread") or 0) == root) for r in later)


def unanswered(ws, me: str, older_s: float = OWED_AFTER_S) -> list[dict[str, Any]]:
    """Direct questions, tasks, handoffs and blocked notices to ``me`` older than ``older_s`` that ``me`` never answered; a request to a subagent is addressed to its own name and is not ``me``'s."""
    now = ws.clock()
    rows = [r for r in ws.bus.log.after(0) if r["kind"] == "msg" and r["ts"] > now - LOOKBACK_S]
    owed = []
    for position, row in enumerate(rows):
        if (row["to"] != me or row["from"] == me or row["type"] not in URGENT or not ws.bus.live(row)
                or now - row["ts"] < older_s):
            continue
        if not _answered(row, rows[position + 1:], me):
            owed.append({"seq": row["seq"], "from": row["from"], "type": row["type"],
                         "age_s": now - row["ts"], "subject": data_line(row["subject"] or row["body"], 60)})
    return owed


def age(seconds: float) -> str:
    """A compact age such as 14m, 3h or 2d."""
    return f"{int(seconds // 86400)}d" if seconds >= 86400 else f"{int(seconds // 3600)}h" if seconds >= 3600 \
        else f"{int(seconds // 60)}m"


def owed_lines(owed: list[dict[str, Any]]) -> list[str]:
    """One line per unanswered request, the oldest first, at most SHOWN of them."""
    lines = [f"[{o['seq']}] {o['type']} from {o['from']}, unanswered {age(o['age_s'])}: {o['subject']}"
             for o in sorted(owed, key=lambda o: -o["age_s"])[:SHOWN]]
    return [*lines, f"(+{len(owed) - SHOWN} more)"] if len(owed) > SHOWN else lines


def attention(ws, me: str, announcements: int) -> str:
    """The text a coordinator is shown: what it owes an answer and how many announcements are new; empty when neither."""
    owed = unanswered(ws, me)
    if not owed and not announcements:
        return ""
    lines = [f"Unanswered for you ({len(owed)}); answer with `ml-stack-workspace send AGENT answer TEXT --reply-to SEQ`:",
             *owed_lines(owed)] if owed else []
    if announcements:
        lines.append(f"{announcements} new announcements (`ml-stack-workspace inbox` shows them).")
    return fence("\n".join(lines), "workspace:attention", "requests written by agents").text


def workers(ws, me: str, now: float) -> dict[str, dict[str, Any]]:
    """The newest announcement of every live subagent below ``me`` not yet done and heard from within ACTIVE_S."""
    below = set(ws.registry.descendants(me, live=True))
    last: dict[str, dict[str, Any]] = {}
    for r in ws.bus.log.after(0):
        if r["kind"] == "msg" and r["to"] == ANNOUNCE and r["from"] in below and r["ts"] > now - ACTIVE_S:
            last[r["from"]] = r
    return {name: row for name, row in last.items() if row["type"] != "done"}


def status_lines(ws, me: str) -> list[str]:
    """One screen: active subagents, their last announcement age and claims, then what is unanswered for ``me``."""
    now = ws.clock()
    held = ws.claims.listing()
    lines = []
    for name, row in sorted(workers(ws, me, now).items(), key=lambda kv: -kv[1]["ts"]):
        mine = [f"{c['kind']}:{c['key']}" for c in held if c["owner"] == name]
        lines.append(f"{name}: {row['type']} {age(now - row['ts'])} ago"
                     f"{', claims ' + ', '.join(mine) if mine else ''}: {data_line(row['body'], 80)}")
    owed = unanswered(ws, me)
    return [f"Active subagents ({len(lines)}):", *(lines or ["(none)"]),
            f"Unanswered for {me} ({len(owed)}):", *(owed_lines(owed) or ["(none)"])]
