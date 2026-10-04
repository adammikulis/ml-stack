"""What a coding harness's tool call is allowed to do under a session role.

`decide` takes the role, the tool's name and its arguments and returns allow, deny or ask. The
destructive-action classifier labels the call; the role decides what each label gets. A call
the classifier cannot read is ``unsure`` and asks. Text naming the launcher's session files, the
state root or an action only a person can take is denied in every role.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ml_stack.chatpolicy import APPROVE_FIRST, PLAN_AND_GO, READ_ONLY, refusal_for
from ml_stack.guard.destructive import classify, reason_text
from ml_stack.interventions import Call

__all__ = ["CATALOG", "SHELL_TOOLS", "Decision", "decide"]

CATALOG: dict[str, str] = {
    **dict.fromkeys(("Read", "Glob", "Grep", "LS", "NotebookRead", "TodoWrite", "TaskList", "TaskGet",
                     "ToolSearch", "BashOutput", "read_file", "list_dir", "grep_files", "update_plan"), "safe"),
    **dict.fromkeys(("Edit", "Write", "MultiEdit", "NotebookEdit", "apply_patch", "WebFetch", "WebSearch"), "reversible"),
}
"""Labels for the harness tools whose effect is known from the name alone."""

SHELL_TOOLS = frozenset({"Bash", "shell", "local_shell", "exec_command", "container.exec", "shell_command"})
"""Tools whose ``command`` argument is a shell line."""


@dataclass(frozen=True, slots=True)
class Decision:
    """``action`` is allow, deny or ask; ``kind`` is the request kind an ask raises."""

    action: str
    label: str
    reason: str
    kind: str = "tool_call"


def _denied(label: str, reason: str) -> Decision:
    return Decision("deny", label, reason)


def _mentions(args: dict[str, Any] | None, words: Sequence[str]) -> str:
    text = json.dumps(args, ensure_ascii=False, default=str) if args is not None else ""
    return next((w for w in words if w and w in text), "")


def _shell_line(name: str, args: dict[str, Any] | None) -> str:
    if name not in SHELL_TOOLS or not args:
        return ""
    line = args.get("command", args.get("cmd", ""))
    return " ".join(map(str, line)) if isinstance(line, list) else str(line)


def decide(role: str, name: str, args: dict[str, Any] | None, *, roots: Sequence[str] = (),
           protected: Sequence[str] = ()) -> Decision:
    """The decision for one tool call. ``protected`` are path strings no call may name."""
    if hit := _mentions(args, protected):
        return _denied("destructive", f"the call names {hit}, which belongs to the launcher")
    if refusal := refusal_for(_shell_line(name, args)):
        return _denied("destructive", f"{refusal[0]} is for a person; run `{refusal[1]}` yourself")
    verdict = classify(Call(name, args), roots=roots, catalog=CATALOG)
    why = reason_text(verdict)
    if verdict.label == "safe":
        return Decision("allow", "safe", why)
    if role == READ_ONLY or role not in (APPROVE_FIRST, PLAN_AND_GO):
        return _denied(verdict.label, f"the read-only role runs no acting call: {why}")
    if verdict.label in ("destructive", "unsure"):
        return Decision("ask", verdict.label, why, "tool_call_destructive")
    if role == APPROVE_FIRST:
        return Decision("ask", verdict.label, why)
    return Decision("allow", verdict.label, why)
