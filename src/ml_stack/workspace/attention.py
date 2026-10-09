"""What a coordinating agent owes the board: unanswered direct requests, live labelled workers, a helper's own messages."""

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


def addressed_to_label(row: dict[str, Any], label: str, sent: set[int]) -> bool:
    """Whether a direct message is meant for the helper called ``label``: it opens with `@label` or `[label]`, or replies into that helper's own message."""
    head = f"{row['subject']} {row['body']}".lstrip().lower()
    if head.startswith((f"@{label}", f"[{label}]")):
        return True
    return bool(sent & {int(row.get("reply_to") or 0), int(row.get("thread") or 0)})


def helper_messages(ws, me: str, label: str) -> list[dict[str, Any]]:
    """Unread direct messages to ``me`` that are for helper ``label``, oldest first."""
    sent = {int(r["seq"]) for r in ws.bus.outbox(me, 1 << 30) if r.get("label") == label}
    return [r for r in ws.bus.inbox(me, limit=1 << 30) if addressed_to_label(r, label, sent)]


def _answered(row: dict[str, Any], later: list[dict[str, Any]], me: str) -> bool:
    root = int(row.get("thread") or row["seq"])
    return any(_mine(r, me) and (r["to"] == row["from"] or int(r.get("reply_to") or 0) == row["seq"]
                                 or int(r.get("thread") or 0) == root) for r in later)


def unanswered(ws, me: str, older_s: float = OWED_AFTER_S) -> list[dict[str, Any]]:
    """Direct questions, tasks, handoffs and blocked notices to ``me`` older than ``older_s`` that ``me`` never answered; those meant for a helper are the helper's, not ``me``'s."""
    now = ws.clock()
    rows = [r for r in ws.bus.log.after(0) if r["kind"] == "msg" and r["ts"] > now - LOOKBACK_S]
    labels = {r["label"] for r in rows if _mine(r, me) and r.get("label")}
    helped = {int(r["seq"]) for r in rows if _mine(r, me) and r.get("label")}
    owed = []
    for position, row in enumerate(rows):
        if (row["to"] != me or row["from"] == me or row["type"] not in URGENT or not ws.bus.live(row)
                or now - row["ts"] < older_s):
            continue
        if any(addressed_to_label(row, label, helped) for label in labels):
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


def workers(ws, now: float) -> dict[tuple[str, str], dict[str, Any]]:
    """The newest announcement of every labelled worker not yet done and heard from within ACTIVE_S."""
    last: dict[tuple[str, str], dict[str, Any]] = {}
    for r in ws.bus.log.after(0):
        if r["kind"] == "msg" and r["to"] == ANNOUNCE and r.get("label") and r["ts"] > now - ACTIVE_S:
            last[(r["from"], r["label"])] = r
    return {key: row for key, row in last.items() if row["type"] != "done"}


def status_lines(ws, me: str) -> list[str]:
    """One screen: active labelled workers, their last announcement age and claims, then what is unanswered for ``me``."""
    now = ws.clock()
    held = ws.claims.listing()
    lines = []
    for (owner, label), row in sorted(workers(ws, now).items(), key=lambda kv: -kv[1]["ts"]):
        mine = [f"{c['kind']}:{c['key']}" for c in held if c["owner"] == owner
                and str(c.get("note", "")).startswith(f"[{label}]")]
        lines.append(f"{owner} ({label}): {row['type']} {age(now - row['ts'])} ago"
                     f"{', claims ' + ', '.join(mine) if mine else ''}: {data_line(row['body'], 80)}")
    owed = unanswered(ws, me)
    return [f"Active workers ({len(lines)}):", *(lines or ["(none)"]),
            f"Unanswered for {me} ({len(owed)}):", *(owed_lines(owed) or ["(none)"])]
