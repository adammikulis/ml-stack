"""Credential redaction: the rail that removes tokens, keys and passwords from text."""

from __future__ import annotations

import os
from collections.abc import Mapping

from ml_stack.interventions import Base, Call, Context, Deny, Proceed, Rewrite, Verdict
from ml_stack.redact.secrets import env_secrets, redact

__all__ = ["SecretRail", "redact", "secrets_in"]

def secrets_in(text: str, known: tuple[str, ...] = ()) -> list[str]:
    """The kinds of credential ``text`` holds, empty when it holds none."""
    return redact(text, known)[1]


class SecretRail(Base):
    """Redacts credentials from text entering and leaving the model, and refuses a tool call
    whose arguments carry one."""

    name = "secrets"

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self.known = env_secrets(os.environ if env is None else env)

    def _scrub(self, text: str, where: str) -> Verdict:
        clean, kinds = redact(text, self.known)
        if not kinds:
            return Proceed()
        return Rewrite(clean, f"{where} held {', '.join(sorted(set(kinds)))}", False, self.name)

    def after_tool_call(self, call: Call, result: str, context: Context) -> Verdict:
        return self._scrub(result, f"tool:{call.name}")

    def after_model_call(self, context: Context, reply: object) -> Verdict:
        text = reply if isinstance(reply, str) else str(getattr(reply, "content", "") or "")
        return self._scrub(text, "model")

    def before_tool_call(self, call: Call, context: Context) -> Verdict:
        flat = " ".join(_strings(call.arguments)) if call.arguments else call.raw
        kinds = secrets_in(flat, self.known)
        if not kinds:
            return Proceed()
        return Deny(f"arguments of {call.name} hold {', '.join(sorted(set(kinds)))}", self.name)


def _strings(value: object) -> list[str]:
    """Every string inside a JSON value, keys included."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in (_strings(k) + _strings(v))]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []
