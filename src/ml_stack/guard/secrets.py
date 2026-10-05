"""Credential redaction: the rail that removes tokens, keys and passwords from text."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

from ml_stack.interventions import Base, Call, Context, Deny, Proceed, Rewrite, Verdict
from ml_stack.redact.secrets import PATTERNS

__all__ = ["SecretRail", "redact", "secrets_in"]

MARK = "[REDACTED:{}]"

ENV_NAME = re.compile(r"(?i)(?:key|secret|token|passw|credential|auth)")
MIN_ENV_VALUE = 8


def env_secrets(env: Mapping[str, str]) -> tuple[str, ...]:
    """The values of the variables whose names say they are credentials, longest first."""
    found = {v for k, v in env.items() if ENV_NAME.search(k) and len(v) >= MIN_ENV_VALUE}
    return tuple(sorted(found, key=len, reverse=True))


def redact(text: str, known: tuple[str, ...] = ()) -> tuple[str, list[str]]:
    """``text`` with every credential replaced by a marker, and the kinds that were found."""
    kinds: list[str] = []
    for value in known:
        if value in text:
            text = text.replace(value, MARK.format("env"))
            kinds.append("env")
    for kind, pattern in PATTERNS:
        text, count = pattern.subn(MARK.format(kind), text)
        if count:
            kinds.append(kind)
    return text, kinds


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
