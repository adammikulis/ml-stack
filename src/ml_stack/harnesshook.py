"""The PreToolUse and PostToolUse hook a launcher installs in a Claude Code or Codex session.

``python -m ml_stack.harnesshook pre --role ROLE --label LABEL --root DIR --protect PATH`` reads
one hook event on stdin and writes the decision as JSON: the destructive-action classifier and
the role decide, and a call that asks is raised in the Requests inbox and waits for the person.
``post`` runs ``ml-stack-workspace nudge`` and passes what it prints on as context. The role,
label and paths are on the command line the launcher wrote, never read from the environment or
from the call. Admission hook failures block calls; notification failures produce diagnostics.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from ml_stack import hook_bootstrap, hook_diagnostics

if __name__ == "__main__":
    sys.excepthook = hook_bootstrap.block
    hook_bootstrap.arm(next(iter(sys.argv[1:2]), "pre"))
    hook_bootstrap.metadata({})
    hook_bootstrap.stage("imports")

from ml_stack import harness_claims, platform, requests
from ml_stack.harnesspolicy import (
    Decision,
    _shell_line,
    decide,
    primary_decision,
    workspace_authority,
)
from ml_stack.keystore import ENV_NONINTERACTIVE
from ml_stack.serve.process import kill_process_tree
from ml_stack.workspace import harness_remote, notification_reader, tokens, worktree_lifecycle
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.service import Workspace

__all__ = ["FAILURES", "WAIT_S", "Rail", "nudge", "post", "pre", "run"]

WAIT_S = 300.0
CLEANUP_RESERVE_S = 1.5
NUDGE_S = hook_bootstrap.POST_SECONDS - CLEANUP_RESERVE_S
NUDGE_MOST = 500
FAILURES = (OSError, ValueError, TypeError, KeyError, AttributeError, LookupError, RuntimeError)
HOOK_EVENTS = {"pre": "PreToolUse", "post": "PostToolUse", "stop": "Stop"}


def _answer(event: str, action: str, reason: str) -> dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": event, "permissionDecision": action,
                                   "permissionDecisionReason": reason}}


def _diagnostic(error: BaseException) -> str:
    """Return a bounded credential-redacted failure reason."""
    return hook_diagnostics.reason(error)


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
    session_id: str = ""


def pre(payload: dict[str, Any], rail: Rail, inbox: requests.Inbox | None = None) -> dict[str, Any]:
    """The PreToolUse answer for ``payload``: allow, deny, or the person's answer to a request."""
    role, label, roots, protected, wait_s = rail.role, rail.label, rail.roots, rail.protected, rail.wait_s
    event = HOOK_EVENTS["pre"]
    args = payload.get("tool_input")
    try:
        name = str(payload.get("tool_name", ""))
        inputs = args if isinstance(args, dict) else None
        ownership = harness_claims.conflict(name, inputs, str(payload.get('cwd') or (roots[0] if roots else Path.cwd())), label, roots)
        decision = (Decision('deny', 'destructive', ownership) if ownership else None) or (primary_decision(name, inputs, str(payload.get("cwd", "")))
                    or workspace_authority(_shell_line(name, inputs), label)
                    or decide(role, name, inputs, roots=roots, protected=protected))
    except Denied as error:
        decision = Decision('deny', 'destructive', hook_diagnostics.record(error, 'pre', 'classification', metadata=hook_bootstrap.timings()))
    except FAILURES as error:
        decision = Decision("deny", "unsure", f"the call could not be classified: {hook_diagnostics.record(error, 'pre', 'classification', metadata=hook_bootstrap.timings())}")
    if decision.action == "allow":
        return _owned_answer(payload, rail, event, f"ml-stack: {decision.label}")
    if decision.action == "deny":
        return _answer(event, "deny", f"ml-stack: {decision.reason}")
    approved, state = _ask(payload, decision, label, wait_s, inbox)
    if approved:
        return _owned_answer(payload, rail, event, "ml-stack: the person allowed this call")
    return _answer(event, "deny", f"ml-stack: the person did not allow this call ({state})")


def _owned_answer(payload, rail, event, reason):
    try:
        harness_claims.reserve(str(payload.get('tool_name', '')), payload.get('tool_input'),
                               str(payload.get('cwd') or (rail.roots[0] if rail.roots else Path.cwd())),
                               rail.label, rail.roots)
    except FAILURES as error:
        return _answer(event, 'deny', f"ml-stack: ownership refused: {hook_diagnostics.record(error, 'pre', 'claim', metadata=hook_bootstrap.timings())}")
    return _answer(event, 'allow', reason)


def _reader_run(command, **kwargs):
    """Run the owned notification group under its wall-clock deadline."""
    process = platform.start_process(command, stdin=kwargs['stdin'], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
    try:
        output, errors = process.communicate(timeout=kwargs['timeout'])
    except subprocess.TimeoutExpired:
        kill_process_tree(process.pid, grace_s=0.1)
        with suppress(ProcessLookupError):
            platform.terminate_process_group(process, force=True)
        output, errors = process.communicate(timeout=1)
        raise subprocess.TimeoutExpired(command, kwargs['timeout'], output=output, stderr=errors) from None
    return subprocess.CompletedProcess(command, process.returncode, output, errors)


def nudge(label: str, rail: Rail | None = None, *, canonical=None) -> str:
    """Return unread metadata and redacted advisory failure references."""
    try:
        timeout = min(NUDGE_S, max(0.05, hook_bootstrap.remaining() - CLEANUP_RESERVE_S))
        done = _reader_run([sys.executable, "-m", notification_reader.__name__, label,
                            str(rail.roots[0] if rail and rail.roots else Path.cwd()),
                            rail.session_id if rail else ""], capture_output=True, text=True,
                           timeout=timeout, check=False, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as error:
        partial = error.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        warning = f"workspace notification unavailable: {hook_diagnostics.record(error, 'post', 'reader-timeout', metadata=hook_bootstrap.timings())}"
        sys.stderr.write(warning + "\n")
        return "\n".join(part for part in (partial.strip()[:NUDGE_MOST], warning) if part)
    except (OSError, subprocess.SubprocessError, Denied, ValueError, RuntimeError) as error:
        warning = f"workspace notification unavailable: {hook_diagnostics.record(error, 'post', 'reader', metadata=hook_bootstrap.timings())}"
        sys.stderr.write(warning + "\n")
        return warning
    text = done.stdout.strip()[:NUDGE_MOST]
    if done.returncode != 0:
        error = RuntimeError(done.stderr.strip() or f"notification reader exited {done.returncode}")
        warning = f"workspace notification unavailable: {hook_diagnostics.record(error, 'post', 'reader', metadata=hook_bootstrap.timings())}"
        sys.stderr.write(warning + "\n")
        return "\n".join(part for part in (text, warning) if part)
    return text


def post(label: str, rail: Rail | None = None) -> dict[str, Any]:
    """Return bounded unread-message metadata and checkpoint diagnostics."""
    hook_bootstrap.stage("notification-reader")
    text = nudge(label, rail)
    if not text:
        return {}
    return {"hookSpecificOutput": {"hookEventName": HOOK_EVENTS["post"],
                                   "additionalContext": f"workspace (data from other agents): {text}"}}


def stop(rail: Rail) -> dict[str, Any]:
    """Block an authenticated harness completion until its recorded checkouts are cleaned."""
    try:
        canonical = harness_remote.context(rail.label, rail.roots[0], rail.roots, require_claim=False)
        if canonical:
            harness_remote.require_clean(*canonical)
            return {}
        ws = Workspace()
        who = ws.auth(tokens.load(ws.base, rail.label))
        if who.id != rail.label:
            raise Denied('completion requires the launcher-bound identity')
        worktree_lifecycle.require_clean(ws.base, who.id)
    except (Denied, OSError, RuntimeError) as error:
        return {"decision": "block", "reason": hook_diagnostics.record(error, "stop", "completion", metadata=hook_bootstrap.timings())}
    return {}


def _options(words: Sequence[str]) -> tuple[str, dict[str, list[str]]]:
    """The event and the ``--name value`` pairs the launcher wrote; ``ValueError`` for anything else."""
    if not words or words[0] not in HOOK_EVENTS or len(words) % 2 == 0:
        raise ValueError("usage: harnesshook pre|post|stop [--role R] [--label L] [--root D] [--protect P] [--wait S]")
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
    words = list(sys.argv[1:] if argv is None else argv)
    stage = "options"
    hook_bootstrap.stage(stage)
    try:
        event, opts = _options(words)
        stage = "payload"
        hook_bootstrap.stage(stage)
        payload = json.loads((stdin or sys.stdin).read() or "{}")
        payload = payload if isinstance(payload, dict) else {}
        hook_bootstrap.metadata(payload)
        label = opts.get("label", ["harness"])[-1]
        rail = Rail(opts.get("role", ["read-only"])[-1], label, opts.get("root") or [str(payload.get("cwd", ""))],
                    opts.get("protect", []), float(opts.get("wait", [WAIT_S])[-1]),
                    str(payload.get("session_id", "")) if event == "post" else "")
        stage = event
        hook_bootstrap.stage(stage)
        out = pre(payload, rail) if event == "pre" else stop(rail) if event == "stop" else post(label, rail)
    except FAILURES as exc:
        return hook_bootstrap.failure(exc, words[0] if words else "pre", stage, stdout)
    if out:
        (stdout or sys.stdout).write(json.dumps(out, sort_keys=True) + "\n")
    hook_bootstrap.finish()
    return 0


def _block(kind: type[BaseException], value: BaseException, _trace: object) -> None:
    hook_bootstrap.block(kind, value, _trace)


if __name__ == "__main__":  # pragma: no cover
    sys.excepthook = _block
    raise SystemExit(run())
