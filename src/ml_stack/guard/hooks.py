"""The guard as Claude Agent SDK hooks, for the harness whose tool loop runs inside Claude Code."""

from __future__ import annotations

from typing import Any

from ml_stack.guard.policy import ToolPolicyRail
from ml_stack.guard.secrets import SecretRail
from ml_stack.guard.untrusted import NOTICE, UntrustedRail
from ml_stack.interventions import Call, Run

__all__ = ["CHANGING", "MAX_TURNS", "sdk_guard", "sdk_hooks"]

CHANGING = frozenset({"Bash", "Write", "Edit", "MultiEdit", "NotebookEdit", "WebFetch"})
"""Claude Code tools that change files, run commands or reach the network."""

READS_OUTSIDE = frozenset({"WebFetch", "WebSearch", "Read", "Grep", "Glob", "BashOutput"})
MAX_TURNS = 50


def sdk_guard() -> list[Any]:
    """The built-in rails for a tool set the guard has no schemas for."""
    return [UntrustedRail(), SecretRail(), ToolPolicyRail(open_world=True)]


def _decision(kind: str, reason: str) -> dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": kind,
                                   "permissionDecisionReason": reason}}


def sdk_hooks(run: Run) -> dict[str, list[Any]]:
    """``PreToolUse`` and ``PostToolUse`` hook entries that put the interventions of ``run`` in
    front of every tool the SDK runs. A changing tool after outside text asks the person; a
    denial blocks."""
    from claude_agent_sdk import HookMatcher

    async def before(data: dict[str, Any], _id: str | None, _ctx: Any) -> dict[str, Any]:
        args = data.get("tool_input")
        call = Call(str(data.get("tool_name") or ""), args if isinstance(args, dict) else None)
        gate = await run.before_tool(call)
        if not gate.allowed:
            return _decision("deny", gate.text)
        if run.tainted and call.name in CHANGING:
            return _decision("ask", "text from outside the person was read; confirm this change")
        return {}

    async def after(data: dict[str, Any], _id: str | None, _ctx: Any) -> dict[str, Any]:
        name = str(data.get("tool_name") or "")
        if name not in READS_OUTSIDE:
            return {}
        shown = await run.after_tool(Call(name), str(data.get("tool_response") or ""))
        if not run.tainted:
            return {}
        why = getattr(shown.verdict, "reason", "")
        return {"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                       "additionalContext": f"{NOTICE} ({why})"}}

    return {"PreToolUse": [HookMatcher(matcher=None, hooks=[before])],
            "PostToolUse": [HookMatcher(matcher=None, hooks=[after])]}
