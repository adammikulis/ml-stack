"""Codex thread identities for shared project connections."""

import hashlib
import os

from ml_stack.workspace.identity import Denied


def current() -> str:
    """Return the current Codex thread's private connection slot."""
    thread = os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_SESSION_ID", "")
    if not thread:
        return ""
    if len(thread) > 256 or any(ord(char) < 33 or ord(char) > 126 for char in thread):
        raise Denied("the Codex session identifier is invalid")
    return hashlib.sha256(thread.encode()).hexdigest()[:32]


def name(agent: str) -> str:
    """Return a distinct Codex agent name for the current thread."""
    slot = current() if agent == "codex" else ""
    return f"codex-{slot}" if slot else agent


def connection(configured: dict) -> dict:
    """Return this thread's saved connection without changing the root authority."""
    slot = current()
    sessions = configured.get("sessions", {})
    if not isinstance(sessions, dict):
        raise Denied("the saved Codex project sessions are invalid")
    saved = sessions.get(slot) if slot else None
    if saved is None:
        return configured
    if (not isinstance(saved, dict) or saved.get("session") != slot
            or not saved.get("agent") or saved.get("local_agent") != "codex"
            or any(saved.get(key) != configured.get(key) for key in ("host", "project_id"))):
        raise Denied("the saved Codex project session is invalid")
    return saved


def owner(value: str) -> str:
    """Return a readable shared Board claim owner."""
    parts = value.split(":", 2)
    if len(parts) == 3 and parts[0] == "canonical" and len(parts[1]) == 32:
        return f"{parts[2]} on shared project Board"
    return value
