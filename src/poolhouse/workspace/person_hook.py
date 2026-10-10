"""The harness hook handlers that record what the person said: UserPromptSubmit, AskUserQuestion answers and session boundaries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from poolhouse.sentinel.events import EventLog
from poolhouse.workspace import (
    person_ancestry,
    person_intent,
    person_record,
    person_release,
    person_store,
    person_targets,
    person_transcript,
)

__all__ = ["STALE_S", "on_answer", "on_boundary", "on_event", "on_prompt", "valid_session"]

STALE_S = 30 * 60


def valid_session(value: Any) -> bool:
    """Whether ``value`` is a plausible native session id."""
    return isinstance(value, str) and 0 < len(value) <= 256 and all(32 < ord(c) < 127 for c in value)


def attended(environ: dict[str, str] | None = None) -> bool:
    """Whether a person is at the session: false for non-interactive runs."""
    env = os.environ if environ is None else environ
    return not env.get("POOLHOUSE_NONINTERACTIVE") and env.get("CLAUDE_CODE_SESSION_ATTENDED", "true").lower() not in (
        "0", "false", "no")


def _context(event: str, text: str) -> dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}


def _stale(turn: person_transcript.Turn) -> bool:
    gap = turn.prompt_ts - turn.proposal_ts
    return turn.compacted or turn.asked or not turn.proposal_ts or gap > STALE_S or gap < 0


def _bind(log: EventLog, session: str) -> None:
    """Record the harness process this hook runs under as the one running ``session``."""
    parent = person_ancestry.harness_parent()
    if parent is None or person_store.bound_sessions(person_store.records(log)).get(parent) == session:
        return
    person_record.record_binding(log, session, parent[0], parent[1])


def on_prompt(event: dict[str, Any], log: EventLog | None = None) -> dict[str, Any] | None:
    """Record a typed prompt; returns the context line for the model, if any. Typed words never authorize."""
    if (event.get("hook_event_name") not in (None, "UserPromptSubmit") or event.get("agent_id")
            or not attended() or not valid_session(event.get("session_id"))):
        return None
    prompt, path, prompt_id = event.get("prompt"), event.get("transcript_path"), event.get("prompt_id")
    if not (isinstance(prompt, str) and isinstance(path, str) and isinstance(prompt_id, str) and prompt_id):
        return None
    session = event["session_id"]
    if person_transcript.path_problem(path, session, os.environ.get("POOLHOUSE_SESSION_ID", "")):
        return None
    turn = person_transcript.wait_for_turn(path, prompt_id, session, prompt)
    if not turn.human:
        return None
    log = log or person_store.open_log()
    cwd = str(event.get("cwd") or str(Path.cwd()))
    checkout = person_targets.inspect(cwd)
    person_record.record_statement(log, person_record.Heard(
        session, checkout.project, cwd, prompt, prompt_id, turn.version, str(Path(path).resolve().parent)))
    _bind(log, session)
    proposal = person_intent.proposal_kinds(turn.proposal) if not _stale(turn) else ()
    reading = person_intent.interpret(prompt, proposal)
    if reading.action == "revoke":
        ended = person_record.revoke_session(log, session, reading.reason)
        return _context("UserPromptSubmit", f"person revoked {len(ended)} authorization(s) (attested by hook)")
    if reading.action == "refuse":
        return _context("UserPromptSubmit", f"not authorizable from chat: {reading.reason}. The owner runs it.")
    if reading.action == "ask":
        return _context("UserPromptSubmit", _approval_request(log, checkout, reading.reason))
    return None


def _approval_request(log: EventLog, checkout: person_targets.Checkout, why: str) -> str:
    found = person_release.facts(checkout)
    if found is None:
        return f"{why}; git cannot show the facts of a release of main, so nothing can be approved"
    yes, no = person_release.options(found)
    return (f"{why}. To release main, ask with AskUserQuestion using exactly this question text (last line "
            f"included) and the options '{yes}' and '{no}':\n{person_release.question(found)}\n"
            f"{person_release.echo_for(log, found)}")


def on_answer(event: dict[str, Any], log: EventLog | None = None) -> dict[str, Any] | None:
    """Record the person's AskUserQuestion answers; the approving answer to a question git still bears out
    becomes a `release-main` authorization."""
    if (event.get("hook_event_name") not in (None, "PostToolUse") or event.get("tool_name") != "AskUserQuestion"
            or event.get("agent_id") or not attended() or not valid_session(event.get("session_id"))):
        return None
    response = event.get("tool_response") or {}
    answers = response.get("answers")
    if not isinstance(answers, dict) or not answers:
        return None
    log = log or person_store.open_log()
    cwd = str(event.get("cwd") or str(Path.cwd()))
    checkout = person_targets.inspect(cwd)
    made = ""
    for question, label in answers.items():
        if not (isinstance(question, str) and isinstance(label, str)):
            continue
        heard = person_record.Heard(event["session_id"], checkout.project, cwd, question)
        row = person_record.record_answer(log, heard, label, bool(response.get("annotations")))
        found = person_release.approved(log, question, label, checkout)
        if found is not None and not made:
            grant = person_record.Grant(person_release.KIND, person_release.target(found), "asked",
                                        row["question_sha256"])
            auth = person_record.record_authorization(log, row, grant)
            made = f"authorization {auth['id']} recorded for release of main to {found.sha}, once (attested by hook)"
    return _context("PostToolUse", made) if made else None


def on_boundary(event: dict[str, Any], name: str, log: EventLog | None = None) -> list[str]:
    """Close the session's live authorizations when it starts again or ends; returns their ids."""
    if event.get("agent_id") or not valid_session(event.get("session_id")):
        return []
    log = log or person_store.open_log()
    ended = person_record.expire_session(log, event["session_id"], name)
    if name == "SessionStart":
        _bind(log, event["session_id"])
    return ended


def on_event(event: dict[str, Any], log: EventLog | None = None) -> dict[str, Any] | None:
    """Route a hook event to its handler by name; session starts and ends close that session's authorizations."""
    name = event.get("hook_event_name")
    if name == "UserPromptSubmit":
        return on_prompt(event, log)
    if name == "PostToolUse":
        return on_answer(event, log)
    if name in ("SessionStart", "SessionEnd"):
        on_boundary(event, str(name), log)
    return None
