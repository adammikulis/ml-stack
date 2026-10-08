"""The harness hook handlers that record what the person said: UserPromptSubmit, AskUserQuestion answers and session boundaries."""

from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Any

from ml_stack.sentinel.events import EventLog
from ml_stack.workspace import person_intent, person_record, person_store, person_targets, person_transcript

__all__ = ["STALE_S", "echo_for", "on_answer", "on_boundary", "on_event", "on_prompt", "valid_session"]

STALE_S = 30 * 60
ANSWER_LABEL = re.compile(r"authorize (?P<kind>[a-z-]+) (?P<target>\S+) (?P<minutes>\d+)m x(?P<uses>\d+) #(?P<tag>[0-9a-f]{12})")


def valid_session(value: Any) -> bool:
    """Whether ``value`` is a plausible native session id."""
    return isinstance(value, str) and 0 < len(value) <= 256 and all(32 < ord(c) < 127 for c in value)


def attended(environ: dict[str, str] | None = None) -> bool:
    """Whether a person is at the session: false for non-interactive runs."""
    env = os.environ if environ is None else environ
    return not env.get("ML_STACK_NONINTERACTIVE") and env.get("CLAUDE_CODE_SESSION_ATTENDED", "true").lower() not in (
        "0", "false", "no")


def echo_for(kind: str, target: str, minutes: int = person_store.DEFAULT_MINUTES, uses: int = 1) -> str:
    """The line a confirmation question must carry: kind, derived target, expiry, uses and a tag over them."""
    tag = hashlib.sha256(f"{kind}|{target}|{minutes}|{uses}".encode()).hexdigest()[:12]
    return f"authorize {kind} {target} {minutes}m x{uses} #{tag}"


def _proposal_hash(kind: str, target: str, ident: str) -> str:
    return hashlib.sha256(f"{kind}|{target}|{ident}".encode()).hexdigest()


def _context(event: str, text: str) -> dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}


def _stale(turn: person_transcript.Turn) -> bool:
    gap = turn.prompt_ts - turn.proposal_ts
    return turn.compacted or turn.asked or not turn.proposal_ts or gap > STALE_S or gap < 0


def on_prompt(event: dict[str, Any], log: EventLog | None = None) -> dict[str, Any] | None:
    """Record a typed prompt and any authorization it speaks; returns the context line for the model, if any."""
    if (event.get("hook_event_name") not in (None, "UserPromptSubmit") or event.get("agent_id")
            or not attended() or not valid_session(event.get("session_id"))):
        return None
    prompt, path, prompt_id = event.get("prompt"), event.get("transcript_path"), event.get("prompt_id")
    if not (isinstance(prompt, str) and isinstance(path, str) and isinstance(prompt_id, str) and prompt_id):
        return None
    turn = person_transcript.wait_for_turn(path, prompt_id, event["session_id"], prompt)
    if not turn.human:
        return None
    log = log or person_store.open_log()
    cwd = str(event.get("cwd") or os.getcwd())
    project = person_targets.project_of(cwd)
    session = event["session_id"]
    statement = person_record.record_statement(log, session_id=session, project=project, cwd=cwd,
                                               prompt=prompt, prompt_id=prompt_id, claude_version=turn.version)
    dev = person_targets.development_branch(cwd)
    proposal = person_intent.proposal_kinds(turn.proposal, dev) if not _stale(turn) else ()
    reading = person_intent.interpret(prompt, proposal, dev)
    return _act(log, reading, statement, turn, cwd, dev)


def _act(log: EventLog, reading: person_intent.Reading, statement: dict[str, Any],
         turn: person_transcript.Turn, cwd: str, dev: str) -> dict[str, Any] | None:
    session = statement["session_id"]
    if reading.action == "revoke":
        ended = person_record.revoke_session(log, session, reading.reason)
        return _context("UserPromptSubmit", f"person revoked {len(ended)} authorization(s) (attested by hook)")
    if reading.action == "refuse":
        return _context("UserPromptSubmit", f"not authorizable from chat: {reading.reason}. The owner runs it.")
    target = person_targets.push_dev_target(cwd) if reading.kind == "push-dev" else ""
    if reading.action == "ask":
        line = echo_for(reading.kind, target) if target else ""
        text = "the person's words do not authorize an action by themselves; ask with AskUserQuestion"
        return _context("UserPromptSubmit", text + (f" and put this line verbatim in the question: {line}" if line else ""))
    if reading.action != "authorize" or not target:
        return None
    ident = turn.proposal_id if reading.how == "reply" else statement["prompt_sha256"]
    made = person_record.record_authorization(log, statement=statement, kind=reading.kind, target=target,
                                              how=reading.how,
                                              proposal_sha256=_proposal_hash(reading.kind, target, ident))
    return _context("UserPromptSubmit", f"authorization {made['id']} recorded for {reading.kind} on {target}, "
                    f"once, until {time.strftime('%H:%M', time.localtime(made['expires']))} (attested by hook)")


def on_answer(event: dict[str, Any], log: EventLog | None = None) -> dict[str, Any] | None:
    """Record the person's AskUserQuestion answers; a verbatim echo answered yes becomes an authorization."""
    if (event.get("hook_event_name") not in (None, "PostToolUse") or event.get("tool_name") != "AskUserQuestion"
            or event.get("agent_id") or not attended() or not valid_session(event.get("session_id"))):
        return None
    answers = (event.get("tool_response") or {}).get("answers")
    if not isinstance(answers, dict) or not answers:
        return None
    log = log or person_store.open_log()
    cwd = str(event.get("cwd") or os.getcwd())
    project = person_targets.project_of(cwd)
    annotated = bool((event.get("tool_response") or {}).get("annotations"))
    made = None
    for question, label in answers.items():
        if not (isinstance(question, str) and isinstance(label, str)):
            continue
        row = person_record.record_answer(log, session_id=event["session_id"], project=project,
                                         question=question, label=label, annotated=annotated)
        made = made or _from_echo(log, row, question, label, cwd)
    return _context("PostToolUse", made) if made else None


def _from_echo(log: EventLog, row: dict[str, Any], question: str, label: str, cwd: str) -> str:
    found = ANSWER_LABEL.search(question)
    if not found or found["kind"] not in person_intent.KINDS or not person_intent.is_affirmative(label):
        return ""
    target = person_targets.push_dev_target(cwd)
    if not target or echo_for(found["kind"], target, int(found["minutes"]), int(found["uses"])) != found[0]:
        return ""
    if int(found["minutes"]) != person_store.DEFAULT_MINUTES or int(found["uses"]) != 1:
        return ""
    made = person_record.record_authorization(log, statement=row, kind=found["kind"], target=target, how="asked",
                                              proposal_sha256=row["question_sha256"])
    return f"authorization {made['id']} recorded for {made['kind']} on {target}, once (attested by hook)"


def on_boundary(event: dict[str, Any], name: str, log: EventLog | None = None) -> list[str]:
    """Close the session's live authorizations when it starts again or ends; returns their ids."""
    if event.get("agent_id") or not valid_session(event.get("session_id")):
        return []
    log = log or person_store.open_log()
    return person_record.expire_session(log, event["session_id"], name)


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
