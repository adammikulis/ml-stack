"""The PreToolUse and PostToolUse hook a launcher installs in a Claude Code or Codex session.

``python -m ml_stack.harnesshook pre --role ROLE --label LABEL --root DIR --protect PATH`` reads
one hook event on stdin and writes the decision as JSON: the destructive-action classifier and
the role decide, and a call that asks is raised in the Requests inbox and waits for the person.
``post`` runs ``ml-stack-workspace nudge`` and passes what it prints on as context. The role,
label and paths are on the command line the launcher wrote, never read from the environment or
from the call. A hook that crashes exits 2, which both harnesses read as a block.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from ml_stack import requests
from ml_stack.harnesspolicy import (
    Decision,
    _shell_line,
    decide,
    primary_decision,
    workspace_authority,
)
from ml_stack.keystore import ENV_NONINTERACTIVE

__all__ = ["FAILURES", "WAIT_S", "Rail", "nudge", "post", "pre", "run"]

WAIT_S = 300.0
NUDGE_S = 5.0
NUDGE_MOST = 500
FAILURES = (OSError, ValueError, TypeError, KeyError, AttributeError, LookupError, RuntimeError)
HOOK_EVENTS = {"pre": "PreToolUse", "post": "PostToolUse"}


def _answer(event: str, action: str, reason: str) -> dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": event, "permissionDecision": action,
                                   "permissionDecisionReason": reason}}


def _summary(args: object) -> str:
    try:
        return json.dumps(args, ensure_ascii=False, default=str)[:200]
    except (TypeError, ValueError):
        return "arguments that could not be read"


def _ask(payload: dict[str, Any], decision: Decision, label: str, wait_s: float,
         inbox: requests.Inbox | None) -> tuple[bool, str]:
    name = str(payload.get("tool_name", ""))
    origin = requests.Origin(label, Path(str(payload.get("cwd", ""))).name, str(payload.get("session_id", "")))
    ask = requests.Ask(decision.kind, f"{name} {_summary(payload.get('tool_input'))}",
                       decision.reason, ("allow-once", "deny"), origin, ttl=wait_s)
    outcome = requests.raise_request(ask, inbox=inbox).wait(timeout=wait_s)
    return outcome.approved, outcome.why or outcome.state


@dataclass(frozen=True, slots=True)
class Rail:
    """What a launcher fixes for a session: the ``role``, the workspace ``label``, the directories
    writes may go to, the ``protected`` paths no call may name and how long a request waits."""

    role: str
    label: str
    roots: Sequence[str] = ()
    protected: Sequence[str] = ()
    wait_s: float = WAIT_S


def pre(payload: dict[str, Any], rail: Rail, inbox: requests.Inbox | None = None) -> dict[str, Any]:
    """The PreToolUse answer for ``payload``: allow, deny, or the person's answer to a request."""
    role, label, roots, protected, wait_s = rail.role, rail.label, rail.roots, rail.protected, rail.wait_s
    event = HOOK_EVENTS["pre"]
    args = payload.get("tool_input")
    try:
        name = str(payload.get("tool_name", ""))
        inputs = args if isinstance(args, dict) else None
        decision = (primary_decision(name, inputs, str(payload.get("cwd", "")))
                    or workspace_authority(_shell_line(name, inputs), label)
                    or decide(role, name, inputs, roots=roots, protected=protected))
    except FAILURES:
        decision = Decision("ask", "unsure", "the call could not be classified", "tool_call_destructive")
    if decision.action == "allow":
        return _answer(event, "allow", f"ml-stack: {decision.label}")
    if decision.action == "deny":
        return _answer(event, "deny", f"ml-stack: {decision.reason}")
    approved, state = _ask(payload, decision, label, wait_s, inbox)
    if approved:
        return _answer(event, "allow", "ml-stack: the person allowed this call")
    return _answer(event, "deny", f"ml-stack: the person did not allow this call ({state})")


def nudge(label: str) -> str:
    """What ``ml-stack-workspace nudge --agent LABEL`` prints, or "" when it prints nothing or
    cannot run."""
    try:
        done = subprocess.run(["ml-stack-workspace", "nudge", "--agent", label], capture_output=True,
                              text=True, timeout=NUDGE_S, check=False, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip()[:NUDGE_MOST] if done.returncode == 0 else ""


def post(label: str) -> dict[str, Any]:
    """The PostToolUse answer: the nudge as context, or nothing."""
    text = nudge(label)
    if not text:
        return {}
    return {"hookSpecificOutput": {"hookEventName": HOOK_EVENTS["post"],
                                   "additionalContext": f"workspace (data from other agents): {text}"}}


def _options(words: Sequence[str]) -> tuple[str, dict[str, list[str]]]:
    """The event and the ``--name value`` pairs the launcher wrote; ``ValueError`` for anything else."""
    if not words or words[0] not in HOOK_EVENTS or len(words) % 2 == 0:
        raise ValueError("usage: harnesshook pre|post [--role R] [--label L] [--root D] [--protect P] [--wait S]")
    found: dict[str, list[str]] = {}
    for key, value in zip(words[1::2], words[2::2], strict=True):
        if key not in ("--role", "--label", "--root", "--protect", "--wait"):
            raise ValueError(f"no option {key}")
        found.setdefault(key[2:], []).append(value)
    return words[0], found


def run(argv: Sequence[str] | None = None, stdin: IO[str] | None = None,
        stdout: IO[str] | None = None) -> int:
    """Read one hook event and write the answer; a failure exits 2, which blocks the call."""
    os.environ[ENV_NONINTERACTIVE] = "1"
    try:
        event, opts = _options(list(sys.argv[1:] if argv is None else argv))
        payload = json.loads((stdin or sys.stdin).read() or "{}")
        payload = payload if isinstance(payload, dict) else {}
        label = opts.get("label", ["harness"])[-1]
        rail = Rail(opts.get("role", ["read-only"])[-1], label, opts.get("root") or [str(payload.get("cwd", ""))],
                    opts.get("protect", []), float(opts.get("wait", [WAIT_S])[-1]))
        out = pre(payload, rail) if event == "pre" else post(label)
    except FAILURES as exc:
        sys.stderr.write(f"ml-stack hook failed, call blocked: {type(exc).__name__}\n")
        return 2
    if out:
        (stdout or sys.stdout).write(json.dumps(out, sort_keys=True) + "\n")
    return 0


def _block(kind: type[BaseException], value: BaseException, _trace: object) -> None:
    sys.stderr.write(f"ml-stack hook failed, call blocked: {kind.__name__}\n")
    os._exit(2)


if __name__ == "__main__":  # pragma: no cover
    sys.excepthook = _block
    raise SystemExit(run())
