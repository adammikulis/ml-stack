"""Reads the Claude Code session transcript to decide whether a prompt was typed by the person, and what the assistant last proposed."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["PINNED_VERSIONS", "Turn", "wait_for_turn"]

PINNED_VERSIONS = ("2.1.293", "2.1.294")
HUMAN_SOURCES = ("typed", "queued")
MAX_BYTES = 256 * 1024 * 1024
LOOKBACK = 400
RETRY_S = 0.1


@dataclass(frozen=True, slots=True)
class Turn:
    """Whether the prompt is the person's, why not when it is not, and the assistant message before it."""

    human: bool
    reason: str = ""
    version: str = ""
    prompt_ts: float = 0.0
    proposal: str = ""
    proposal_id: str = ""
    proposal_ts: float = 0.0
    asked: bool = False
    compacted: bool = False


def _seconds(stamp: Any) -> float:
    try:
        return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and all(isinstance(b, dict) and b.get("type") == "text" for b in content):
        return "".join(str(b.get("text", "")) for b in content)
    return None


def _entries(path: Path) -> list[dict[str, Any]]:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("transcript too large")
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _prompt_entry(entries: list[dict[str, Any]], prompt_id: str) -> int:
    """The index of the user entry that carries the prompt text for ``prompt_id``, or -1."""
    for index, row in enumerate(entries):
        if row.get("type") == "user" and row.get("promptId") == prompt_id and "origin" in row:
            return index
    return -1


def _judge(row: dict[str, Any], session_id: str, prompt: str) -> str:
    """A reason the entry is not a typed prompt of this session, or an empty string."""
    origin = row.get("origin")
    if not isinstance(origin, dict) or origin.get("kind") != "human":
        return f"origin is {origin!r}"
    if row.get("turnOrigin") != "human":
        return f"turnOrigin is {row.get('turnOrigin')!r}"
    if row.get("promptSource") not in HUMAN_SOURCES:
        return f"promptSource is {row.get('promptSource')!r}"
    if row.get("version") not in PINNED_VERSIONS:
        return f"Claude Code version {row.get('version')!r} is not pinned"
    if row.get("sessionId") != session_id or row.get("isSidechain"):
        return "entry belongs to another session or a sidechain"
    text = _text((row.get("message") or {}).get("content"))
    if text is None or hashlib.sha256(text.encode()).digest() != hashlib.sha256(prompt.encode()).digest():
        return "entry text differs from the hook prompt"
    return ""


def _before(entries: list[dict[str, Any]], index: int) -> tuple[str, str, float, bool, bool]:
    """``(text, message id, time, asked, compacted)`` of the assistant message before ``index``."""
    compacted, asked, parts, ident, stamp, seen = False, False, [], "", 0.0, False
    for row in reversed(entries[max(0, index - LOOKBACK):index]):
        if row.get("subtype") == "compact_boundary" or row.get("isCompactSummary"):
            compacted = True
        if row.get("type") != "assistant":
            if seen:
                break
            continue
        seen = True
        message = row.get("message") or {}
        if ident and message.get("id") != ident:
            break
        ident = str(message.get("id") or row.get("uuid") or "")
        stamp = stamp or _seconds(row.get("timestamp"))
        for block in reversed(message.get("content") or []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") == "AskUserQuestion":
                asked = True
            if block.get("type") == "text":
                parts.insert(0, str(block.get("text", "")))
        if not message.get("id"):
            break
    return "\n".join(parts), ident, stamp, asked, compacted


def read_turn(path: Path, prompt_id: str, session_id: str, prompt: str) -> Turn:
    """The `Turn` for ``prompt_id``; not human on a missing entry, an unknown shape or an unpinned version."""
    try:
        entries = _entries(path)
    except (OSError, ValueError) as error:
        return Turn(False, f"transcript unreadable: {error}")
    index = _prompt_entry(entries, prompt_id)
    if index < 0:
        return Turn(False, "no transcript entry for this prompt")
    row = entries[index]
    why = _judge(row, session_id, prompt)
    if why:
        return Turn(False, why, str(row.get("version", "")))
    text, ident, stamp, asked, compacted = _before(entries, index)
    return Turn(True, "", str(row["version"]), _seconds(row.get("timestamp")), text, ident, stamp, asked,
                compacted)


def wait_for_turn(path: str, prompt_id: str, session_id: str, prompt: str, *, seconds: float = 2.0) -> Turn:
    """`read_turn`, retried for up to ``seconds`` because the transcript is written asynchronously."""
    deadline = time.monotonic() + seconds
    where = Path(path)
    while True:
        turn = read_turn(where, prompt_id, session_id, prompt)
        if turn.human or (turn.reason != "no transcript entry for this prompt"
                          and not turn.reason.startswith("transcript unreadable")
                          and not turn.reason.startswith("entry text differs")):
            return turn
        if time.monotonic() >= deadline:
            return turn
        time.sleep(RETRY_S)
