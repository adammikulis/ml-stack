"""The line a hook shows when messages wait, and the Claude Code hook outputs built from it."""

from __future__ import annotations

import getpass
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    """The unread rows for one reader at one moment; only kinds, senders and times are read."""

    me: str
    rows: list[dict[str, Any]]
    now: float

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
            who = SAFE.sub("", f"{r['from']}/{r['label']}" if r.get("label") else str(r["from"]))
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


def output(event: str, waiting: Waiting, stdin: str) -> str:
    """The JSON a Claude Code ``event`` hook prints ("post", "stop" or "prompt"), or ""."""
    if event == "post":
        stamp = _stamp("ml-stack-nudge")
        if waiting.now - _read_int(stamp) < POST_EVERY_S:
            return ""
        _write(stamp, int(waiting.now))
    line = waiting.line()
    if not line:
        return ""
    if event == "stop":
        return _stop(waiting, line, stdin)
    return json.dumps({"hookSpecificOutput": {"hookEventName": NAMES[event], "additionalContext": line}})


def _stop(waiting: Waiting, line: str, stdin: str) -> str:
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
    return json.dumps({"decision": "block", "reason": line})
