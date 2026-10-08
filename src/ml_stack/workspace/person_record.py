"""The writers of the person record. Only the harness hook entry points and `person_auth` import this module."""

from __future__ import annotations

import hashlib
import secrets
import time
from typing import Any

from ml_stack.sentinel.events import EventLog
from ml_stack.sentinel.redaction import redact
from ml_stack.workspace import person_store
from ml_stack.workspace.person_store import ATTESTATION, VERSION

__all__ = ["EXCERPT_CHARS", "SOURCE", "expire_session", "mark_used", "prompt_hash", "record_answer",
           "record_authorization", "record_statement", "revoke_session"]

SOURCE = "harness-hook:UserPromptSubmit"
EXCERPT_CHARS = 160


def prompt_hash(text: str) -> str:
    """The SHA-256 of ``text``."""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _excerpt(text: str) -> str:
    return " ".join(redact(text).split())[:EXCERPT_CHARS]


def _append(log: EventLog, row: dict[str, Any]) -> dict[str, Any]:
    return log.append_record({"version": VERSION, "attestation": ATTESTATION,
                              "ts": round(time.time(), 3), **row})


def record_statement(log: EventLog, *, session_id: str, project: str, cwd: str, prompt: str,
                     prompt_id: str, claude_version: str) -> dict[str, Any]:
    """Append what the person typed: its hash and a redacted excerpt, never the prompt itself."""
    return _append(log, {"type": "statement", "session_id": session_id, "project": project, "cwd": cwd,
                         "prompt_sha256": prompt_hash(prompt), "excerpt": _excerpt(prompt),
                         "prompt_id": prompt_id, "claude_version": claude_version, "source": SOURCE})


def record_answer(log: EventLog, *, session_id: str, project: str, question: str, label: str,
                  annotated: bool) -> dict[str, Any]:
    """Append the person's answer to an AskUserQuestion: the question's hash and excerpt and the chosen label."""
    return _append(log, {"type": "answer", "session_id": session_id, "project": project,
                         "question_sha256": prompt_hash(question), "excerpt": _excerpt(question),
                         "label": _excerpt(label), "annotated": annotated,
                         "source": "harness-hook:PostToolUse:AskUserQuestion"})


def record_authorization(log: EventLog, *, statement: dict[str, Any], kind: str, target: str, how: str,
                         proposal_sha256: str = "", minutes: int = person_store.DEFAULT_MINUTES,
                         uses: int = 1) -> dict[str, Any]:
    """Append an authorization spoken in ``statement``; it expires after ``minutes`` (at most four hours)."""
    minutes = max(1, min(int(minutes), person_store.MAX_MINUTES))
    return _append(log, {"type": "authorization", "id": secrets.token_hex(6),
                         "statement_seq": statement["seq"], "session_id": statement["session_id"],
                         "project": statement["project"], "kind": kind, "target": target,
                         "target_rule": kind, "uses": uses, "how": how,
                         "proposal_sha256": proposal_sha256,
                         "expires": round(time.time() + minutes * 60, 3)})


def mark_used(log: EventLog, auth_id: str, *, by: str, target: str) -> dict[str, Any]:
    """Append one use of an authorization by the agent labelled ``by``."""
    return _append(log, {"type": "transition", "auth_id": auth_id, "state": "used", "by": by,
                         "target": target})


def _end(log: EventLog, rows: list[dict[str, Any]], session_id: str, state: str,
         reason: str) -> list[str]:
    ended = []
    for auth in person_store.authorizations(rows):
        if auth.session_id == session_id and auth.state == "live":
            _append(log, {"type": "transition", "auth_id": auth.id, "state": state, "reason": reason})
            ended.append(auth.id)
    return ended


def revoke_session(log: EventLog, session_id: str, reason: str) -> list[str]:
    """End every live authorization spoken in ``session_id``; returns their ids."""
    return _end(log, person_store.records(log), session_id, "revoked", reason)


def expire_session(log: EventLog, session_id: str, reason: str) -> list[str]:
    """Close every live authorization of a session that ended or began again; returns their ids."""
    return _end(log, person_store.records(log), session_id, "expired", reason)
