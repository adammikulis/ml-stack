"""The writers of the person record. Only the harness hook entry points and `person_auth` import this module."""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Any

from poolhouse.sentinel.events import EventLog
from poolhouse.sentinel.redaction import redact
from poolhouse.workspace import person_store
from poolhouse.workspace.person_store import ATTESTATION, VERSION

__all__ = ["EXCERPT_CHARS", "SOURCE", "Grant", "Heard", "expire_session", "mark_used", "prompt_hash", "record_answer",
           "record_authorization", "record_binding", "record_statement", "revoke_session"]

SOURCE = "harness-hook:UserPromptSubmit"
EXCERPT_CHARS = 80


def prompt_hash(text: str) -> str:
    """The SHA-256 of ``text``."""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _excerpt(text: str) -> str:
    return " ".join(redact(text).split())[:EXCERPT_CHARS]


def _append(log: EventLog, row: dict[str, Any]) -> dict[str, Any]:
    return log.append_record({"version": VERSION, "attestation": ATTESTATION,
                              "ts": round(time.time(), 3), **row})


@dataclass(frozen=True, slots=True)
class Heard:
    """What the hook received: the session, the project, the working directory, the text typed or asked,
    its id and the Claude Code version that wrote it."""

    session_id: str
    project: str
    cwd: str
    text: str
    ident: str = ""
    version: str = ""
    transcript: str = ""


@dataclass(frozen=True, slots=True)
class Grant:
    """What an authorization allows: a kind on a derived target, how it was spoken, its proposal hash,
    its life in minutes (at most four hours) and its uses."""

    kind: str
    target: str
    how: str
    proposal_sha256: str = ""
    minutes: int = person_store.DEFAULT_MINUTES
    uses: int = 1


def record_statement(log: EventLog, heard: Heard) -> dict[str, Any]:
    """Append what the person typed: its hash and a redacted excerpt, never the prompt itself."""
    return _append(log, {"type": "statement", "session_id": heard.session_id, "project": heard.project,
                         "cwd": heard.cwd, "prompt_sha256": prompt_hash(heard.text),
                         "excerpt": _excerpt(heard.text), "prompt_id": heard.ident,
                         "claude_version": heard.version, "transcript_dir": heard.transcript,
                         "source": SOURCE})


def record_answer(log: EventLog, heard: Heard, label: str, annotated: bool) -> dict[str, Any]:
    """Append the person's answer to an AskUserQuestion: the question's hash and excerpt and the chosen label."""
    return _append(log, {"type": "answer", "session_id": heard.session_id, "project": heard.project,
                         "question_sha256": prompt_hash(heard.text), "excerpt": _excerpt(heard.text),
                         "label": _excerpt(label), "annotated": annotated,
                         "source": "harness-hook:PostToolUse:AskUserQuestion"})


def record_authorization(log: EventLog, statement: dict[str, Any], grant: Grant) -> dict[str, Any]:
    """Append an authorization spoken in ``statement``; it expires after ``grant.minutes`` (at most four hours)."""
    minutes = max(1, min(int(grant.minutes), person_store.MAX_MINUTES))
    return _append(log, {"type": "authorization", "id": secrets.token_hex(6),
                         "statement_seq": statement["seq"], "session_id": statement["session_id"],
                         "project": statement["project"], "kind": grant.kind, "target": grant.target,
                         "target_rule": grant.kind, "uses": grant.uses, "how": grant.how,
                         "proposal_sha256": grant.proposal_sha256,
                         "expires": round(time.time() + minutes * 60, 3)})


def mark_used(log: EventLog, auth_id: str, *, by: str, target: str) -> dict[str, Any]:
    """Append one use of an authorization by the agent labelled ``by``."""
    return _append(log, {"type": "transition", "auth_id": auth_id, "state": "used", "by": by,
                         "target": target})


def _end(log: EventLog, session_id: str, state: str, reason: str) -> list[str]:
    ended: list[str] = []
    if not log.path.exists():
        return ended
    with person_store.consume_lock(log):
        for auth in person_store.authorizations(person_store.records(log)):
            if auth.session_id == session_id and auth.state == "live":
                _append(log, {"type": "transition", "auth_id": auth.id, "state": state, "reason": reason})
                ended.append(auth.id)
    return ended


def record_binding(log: EventLog, session_id: str, pid: int, created: float) -> dict[str, Any]:
    """Append that the harness process ``pid`` (started at ``created``) runs ``session_id``."""
    return _append(log, {"type": "binding", "session_id": session_id, "pid": pid, "created": created})


def revoke_session(log: EventLog, session_id: str, reason: str) -> list[str]:
    """End every live authorization spoken in ``session_id``; returns their ids."""
    return _end(log, session_id, "revoked", reason)


def expire_session(log: EventLog, session_id: str, reason: str) -> list[str]:
    """Close every live authorization of a session that ended or began again; returns their ids."""
    return _end(log, session_id, "expired", reason)
