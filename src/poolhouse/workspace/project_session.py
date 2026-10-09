"""Native session identities for shared project connections."""

import hashlib
import os

from poolhouse.workspace.identity import Denied, valid_name


def harness() -> str:
    """Return the native harness providing this process's session context."""
    native = os.environ.get("POOLHOUSE_SESSION_HARNESS", "")
    session = os.environ.get("POOLHOUSE_SESSION_ID", "")
    if bool(native) != bool(session):
        raise Denied("the native session harness and identifier must be paired")
    if native:
        if len(native) > 15 or not valid_name(native):
            raise Denied("the native session harness is invalid")
        return native
    return "codex" if os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_SESSION_ID") else ""


def current(agent: str = "") -> str:
    """Return this native session's private connection slot."""
    native = harness()
    if not native or (agent and agent != native):
        return ""
    session = (os.environ.get("POOLHOUSE_SESSION_ID") if os.environ.get("POOLHOUSE_SESSION_HARNESS")
               else os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_SESSION_ID", ""))
    if len(session) > 256 or any(ord(char) < 33 or ord(char) > 126 for char in session):
        raise Denied("the native session identifier is invalid")
    value = session if native == "codex" else native + "\0" + session
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def name(agent: str) -> str:
    """Return the opaque routing identity for this native session."""
    slot = current(agent)
    return f"{agent}-{slot}" if slot else agent


def connection(configured: dict) -> dict:
    """Return this session's saved connection under the existing root authority."""
    slot = current()
    sessions = configured.get("sessions", {})
    if not isinstance(sessions, dict):
        raise Denied("the saved native project sessions are invalid")
    saved = sessions.get(slot) if slot else None
    if saved is None:
        return configured
    if (not isinstance(saved, dict) or saved.get("session") != slot
            or not saved.get("agent") or saved.get("local_agent") != harness()
            or any(saved.get(key) != configured.get(key) for key in ("host", "project_id"))):
        raise Denied("the saved native project session is invalid")
    return saved


def owner(value: str) -> str:
    """Return a readable shared Board claim owner."""
    parts = value.split(":", 2)
    if len(parts) == 3 and parts[0] == "board" and len(parts[1]) == 32:
        return f"{parts[2]} on shared project Board"
    return value
