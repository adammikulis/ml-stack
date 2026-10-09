"""The line a hook shows when messages wait, and the Claude Code hook outputs built from it."""

from __future__ import annotations

import getpass
import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.workspace import nudge_fence

URGENT = ("question", "task", "handoff", "blocked")
UNCHANGING = frozenset({"status", "blocked", "done"})
STOP_AFTER_S = 120.0
POST_EVERY_S = 20.0
SENDERS_SHOWN = 4
EVENTS = ("post", "stop", "prompt")
NAMES = {"post": "PostToolUse", "prompt": "UserPromptSubmit"}
SAFE = re.compile(r"[^A-Za-z0-9._/-]")


@dataclass(frozen=True, slots=True)
class Waiting:
    """The unread rows for one reader at one moment, and the text of those from senders it may be pushed for."""

    me: str
    rows: list[dict[str, Any]]
    now: float
    messages: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """The reader, the time and each unread row's sequence, sender, kind and time, and the pushable messages."""
        keys = ("seq", "to", "from", "type", "ts")
        return {"me": self.me, "now": self.now, "rows": [{k: r.get(k) for k in keys} for r in self.rows],
                "messages": self.messages}

    @classmethod
    def of(cls, summary: dict[str, Any]) -> Waiting:
        """The `Waiting` a `summary` describes."""
        return cls(str(summary["me"]), list(summary["rows"]), float(summary["now"]),
                   list(summary.get("messages", [])))

    def urgent(self) -> list[dict[str, Any]]:
        """Direct questions, tasks, handoffs and blocked notices addressed to the reader."""
        return [r for r in self.rows if r["to"] == self.me and r.get("type") in URGENT]

    def _kinds(self) -> str:
        counts: dict[str, int] = {}
        for r in self.rows:
            kind = str(r.get("type") or "message")
            counts[kind] = counts.get(kind, 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: (kv[0] not in URGENT, -kv[1], kv[0]))
        return ", ".join(f"{n} {k}{'s' if n > 1 and k not in UNCHANGING else ''}" for k, n in ordered)

    def _senders(self) -> str:
        seen: list[str] = []
        for r in self.rows:
            who = SAFE.sub("", str(r["from"]))
            if who and who not in seen:
                seen.append(who)
        more = f" +{len(seen) - SENDERS_SHOWN} more" if len(seen) > SENDERS_SHOWN else ""
        return ", ".join(seen[:SENDERS_SHOWN]) + more

    def line(self) -> str:
        """The one-line summary, or "" when nothing waits."""
        if not self.rows:
            return ""
        age = age_text(self.now - min(float(r["ts"]) for r in self.rows))
        head = (f"workspace: {len(self.rows)} waiting for you ({self._kinds()}; "
                f"from {self._senders()}; oldest {age})")
        urgent = sorted({r["type"] for r in self.urgent()}, key=URGENT.index)
        if not urgent:
            return f"{head}; run inbox"
        return (f"{head}. A direct {' or '.join(urgent)} is waiting on you: run "
                f"ml-stack-workspace inbox now and answer it")


def age_text(seconds: float) -> str:
    """``seconds`` as 40s, 5m or 3h12m."""
    s = max(int(seconds), 0)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def _stamp(name: str) -> Path:
    return Path(tempfile.gettempdir()) / f"{name}.{getpass.getuser()}"


def _read_int(path: Path) -> int:
    try:
        return int(float(path.read_text().strip() or 0))
    except (OSError, ValueError):
        return 0


def _write(path: Path, value: float) -> None:
    try:
        path.write_text(str(value))
    except OSError:
        return


def _seen_file(me: str) -> Path:
    return _stamp(f"ml-stack-nudge-seen.{me}")


def _delivery(waiting: Waiting, rows: list[dict[str, Any]], seen: int) -> tuple[str, int]:
    """The text pushed for ``rows``: the summary line, the pushable messages inside one nonce fence
    within the caps, a count with the sequence numbers of the rest, and the standing notice; with
    the highest sequence number covered. Rows at or below ``seen`` were delivered before."""
    pushable = {int(m["seq"]): m for m in waiting.messages}
    fresh = [r for r in rows if int(r["seq"]) > seen]
    blocks, shown, used = [], set(), 0
    for row in fresh:
        message = pushable.get(int(row["seq"]))
        if message is None or len(blocks) == nudge_fence.MESSAGES:
            continue
        text = nudge_fence.block(message)
        if blocks and used + len(text) > nudge_fence.DELIVERY_CHARS:
            break
        blocks.append(text)
        shown.add(int(row["seq"]))
        used += len(text)
    rest = [int(r["seq"]) for r in fresh if int(r["seq"]) not in shown]
    parts = [waiting.line(), *([nudge_fence.fenced(blocks)] if blocks else [])]
    if rest:
        listed = ", ".join(str(n) for n in rest[:nudge_fence.SEQS_LISTED])
        more = ", ..." if len(rest) > nudge_fence.SEQS_LISTED else ""
        parts.append(f"{len(rest)} more not shown (seq {listed}{more}); run ml-stack-workspace inbox")
    parts.append(nudge_fence.NOTICE)
    return "\n".join(parts), max(int(r["seq"]) for r in rows)


def output(event: str, waiting: Waiting, stdin: str) -> str:
    """The JSON a Claude Code ``event`` hook prints ("post", "stop" or "prompt"), or "". Messages not
    delivered before go to the model as `additionalContext` and to the person as `systemMessage`;
    a delivered message is not delivered again."""
    if event == "post":
        stamp = _stamp("ml-stack-nudge")
        if waiting.now - _read_int(stamp) < POST_EVERY_S:
            return ""
        _write(stamp, int(waiting.now))
    seen = _read_int(_seen_file(waiting.me))
    if event == "stop":
        return _stop(waiting, stdin, seen)
    rows = [r for r in waiting.rows if int(r["seq"]) > seen]
    if not rows:
        return ""
    text, top = _delivery(waiting, rows, seen)
    _write(_seen_file(waiting.me), top)
    return json.dumps({"hookSpecificOutput": {"hookEventName": NAMES[event], "additionalContext": text},
                       "systemMessage": text})


def _stop(waiting: Waiting, stdin: str, seen: int) -> str:
    try:
        active = bool(json.loads(stdin or "{}").get("stop_hook_active"))
    except (ValueError, AttributeError):
        active = False
    urgent = waiting.urgent()
    old = [r for r in urgent if waiting.now - float(r["ts"]) >= STOP_AFTER_S]
    state = _stamp(f"ml-stack-nudge-stop.{waiting.me}")
    top = max((int(r["seq"]) for r in urgent), default=0)
    if active or not old or top <= _read_int(state):
        return ""
    _write(state, top)
    text, last = _delivery(waiting, urgent, seen)
    _write(_seen_file(waiting.me), max(seen, last))
    return json.dumps({"decision": "block", "reason": text, "systemMessage": text})
