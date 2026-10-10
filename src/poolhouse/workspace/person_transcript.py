"""Reads the Claude Code session transcript to decide whether a prompt was typed by the person, and what the assistant last proposed."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from poolhouse import home

__all__ = ["PINNED_VERSIONS", "Turn", "claude_projects", "path_problem", "wait_for_turn"]

PINNED_VERSIONS = ("2.1.293", "2.1.294")
HUMAN_SOURCES = ("typed", "queued")
TAIL_BYTES = 8 * 1024 * 1024
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


def claude_projects() -> Path:
    """The folder Claude Code keeps session transcripts in, resolved."""
    root = os.environ.get("CLAUDE_CONFIG_DIR") or str(home.user_home() / ".claude")
    return (Path(root).expanduser() / "projects").resolve()


def path_problem(path: str, session_id: str, env_session: str = "") -> str:
    """Why ``path`` is not the transcript of ``session_id`` under the Claude projects folder, or an empty
    string: it must be an owned regular file, not a link, named ``<session_id>.jsonl``, and the session the
    calling process exports, when there is one, must be the same."""
    where = Path(path)
    try:
        info = where.lstat()
        inside = where.resolve().is_relative_to(claude_projects())
    except OSError as error:
        return f"transcript unreadable: {error}"
    if env_session and env_session != session_id:
        return "session differs from the calling process's session"
    if where.is_symlink() or where.resolve() != Path(os.path.normpath(where.absolute())) or not inside:
        return "transcript is not a plain file under the Claude projects folder"
    if info.st_uid != os.getuid() or where.name != session_id + ".jsonl":
        return "transcript is not this user's file for this session"
    return ""


def _entries(path: Path) -> list[dict[str, Any]]:
    """The entries in the last ``TAIL_BYTES`` of the transcript."""
    with path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        handle.seek(max(0, size - TAIL_BYTES))
        data = handle.read()
    lines = data.splitlines()
    if size > TAIL_BYTES and lines:
        lines = lines[1:]
    out = []
    for line in lines:
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
