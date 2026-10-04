"""The PreToolUse and PostToolUse hook a launcher installs in a Claude Code or Codex session.

``python -m ml_stack.harnesshook pre --role ROLE --label LABEL --root DIR --protect PATH`` reads
one hook event on stdin and writes the decision as JSON: the destructive-action classifier and
the role decide, and a call that asks is raised in the Requests inbox and waits for the person.
``post`` runs ``ml-stack-workspace nudge`` and passes what it prints on as context. The role,
label and paths are on the command line the launcher wrote, never read from the environment or
from the call. A hook that crashes exits 2, which both harnesses read as a block.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from ml_stack import requests
from ml_stack.harnesspolicy import Decision, decide
from ml_stack.keystore import ENV_NONINTERACTIVE

__all__ = ["WAIT_S", "Rail", "main", "nudge", "post", "pre"]

WAIT_S = 300.0
NUDGE_S = 5.0
NUDGE_MOST = 500
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
        decision = decide(role, str(payload.get("tool_name", "")), args if isinstance(args, dict) else None,
                          roots=roots, protected=protected)
    except Exception:  # noqa: BLE001 - a call that cannot be classified is asked about
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


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ml_stack.harnesshook", allow_abbrev=False)
    ap.add_argument("event", choices=sorted(HOOK_EVENTS))
    ap.add_argument("--role", default="read-only")
    ap.add_argument("--label", default="harness")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--protect", action="append", default=[])
    ap.add_argument("--wait", type=float, default=WAIT_S)
    return ap


def main(argv: Sequence[str] | None = None, stdin: IO[str] | None = None,
         stdout: IO[str] | None = None) -> int:
    args = parser().parse_args(argv)
    os.environ[ENV_NONINTERACTIVE] = "1"
    try:
        payload = json.loads((stdin or sys.stdin).read() or "{}")
        if not isinstance(payload, dict):
            payload = {}
        roots = args.root or [str(payload.get("cwd", ""))]
        rail = Rail(args.role, args.label, roots, args.protect, args.wait)
        out = pre(payload, rail) if args.event == "pre" else post(args.label)
    except Exception as exc:  # noqa: BLE001 - exit 2 blocks; any other failure lets the call through
        sys.stderr.write(f"ml-stack hook failed, call blocked: {type(exc).__name__}\n")
        return 2
    if out:
        (stdout or sys.stdout).write(json.dumps(out, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
