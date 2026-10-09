"""The guard as Claude Agent SDK hooks, for the harness whose tool loop runs inside Claude Code."""

from __future__ import annotations

from typing import Any

from poolhouse.guard.policy import ToolPolicyRail
from poolhouse.guard.secrets import SecretRail
from poolhouse.guard.untrusted import NOTICE, UntrustedRail
from poolhouse.interventions import Call, Confirm, Deny, Run
from poolhouse.taint import TaintRail, claude_code

__all__ = ["sdk_guard", "sdk_hooks"]

READS_OUTSIDE = frozenset({"WebFetch", "WebSearch", "Read", "Grep", "Glob", "BashOutput"})


def sdk_guard() -> list[Any]:
    """The built-in rails for a tool set the guard has no schemas for."""
    return [UntrustedRail(), SecretRail(), ToolPolicyRail(open_world=True),
            TaintRail(claude_code())]


def _decision(kind: str, reason: str) -> dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": kind,
                                   "permissionDecisionReason": reason}}


def sdk_hooks(run: Run) -> dict[str, list[Any]]:
    """``PreToolUse`` and ``PostToolUse`` hook entries that put the interventions of ``run`` in
    front of every tool the SDK runs. A `Deny` blocks; a `Confirm` (the taint rail's, after outside
    text was read) is answered ``ask``, which puts the question to the person."""
    from claude_agent_sdk import HookMatcher

    async def before(data: dict[str, Any], _id: str | None, _ctx: Any) -> dict[str, Any]:
        args = data.get("tool_input")
        call = Call(str(data.get("tool_name") or ""), args if isinstance(args, dict) else None)
        asks: list[str] = []
        for verdict in await run.ask("before_tool_call", call, run.context):
            if isinstance(verdict, Deny):
                return _decision("deny", f"Denied: {verdict.reason}")
            if isinstance(verdict, Confirm):
                asks.append(verdict.question)
        return _decision("ask", "; ".join(asks)) if asks else {}

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
