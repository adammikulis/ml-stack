"""Profiles of a local agent (chat-sized or coding-sized), the context it is served at and admitted
for, and the trim that keeps a long task inside that context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ml_stack.serve import wired

__all__ = ["CHAT", "CODING", "MARKER", "PROFILES", "Profile", "admit", "parse_ctx", "profile", "trim"]

MARKER = "[earlier turns of this task were dropped to stay inside the context]"
KV = "q8_0"
_CTX = re.compile(r"^(\d+)([kKmM]?)$")


@dataclass(frozen=True, slots=True)
class Profile:
    """A context and the per-task caps sized for it."""

    name: str
    ctx: int
    rounds: int
    calls: int
    steps: int
    seconds: float


CHAT = Profile("chat", 0, 12, 30, 24, 600.0)
CODING = Profile("coding", 0, 60, 150, 120, 3600.0)
PROFILES = {p.name: p for p in (CHAT, CODING)}


def profile(name: str) -> Profile:
    """The profile called ``name``; ValueError naming the choices otherwise."""
    if name not in PROFILES:
        raise ValueError(f"profile is one of {', '.join(PROFILES)}")
    return PROFILES[name]


def parse_ctx(text: str) -> int:
    """Tokens from ``32768``, ``32k`` or ``256K`` (k is 1024); 0 for empty text."""
    if not text:
        return 0
    found = _CTX.match(text.strip())
    if not found:
        raise ValueError(f"{text!r} is not a context size such as 32768, 32k or 256k")
    n = int(found.group(1)) * {"": 1, "k": 1024, "m": 1024 * 1024}[found.group(2).lower()]
    if not 2048 <= n <= 4 * 1024 * 1024:
        raise ValueError("the context is between 2048 tokens and 4M")
    return n


def admit(model: str, ctx: int) -> tuple[str, str]:
    """``("", "")`` when ``model`` at ``ctx`` (q8_0 cache, MTP head shared) fits under the memory
    limit now; otherwise the one-line reason with the longest context that does fit and, as the
    hint, the command a person runs to raise the limit. A model the estimator cannot read is let
    through to the broker."""
    try:
        plan = wired.plan(model, wired.Ask(context=ctx, kv=KV))
    except (FileNotFoundError, ValueError, OSError):
        return "", ""
    if plan.enough_now:
        return "", ""
    now = max((r.context for r in plan.table if r.enough_now), default=0)
    gib = plan.need_bytes / wired.GIB
    fit = f"the longest context that fits now is {now // 1024}K" if now else "no listed context fits now"
    return (f"{plan.model} at {ctx // 1024}K needs {gib:.1f} GiB wired and that is over this "
            f"machine's current limit; {fit}",
            f"ml-stack-serve memory --for {plan.model} --ctx {ctx} --kv {KV} --apply")


def _groups(messages: list[dict[str, Any]], start: int) -> list[tuple[int, int]]:
    """Index ranges of the whole turns from ``start``: an assistant message with its tool results."""
    out, i = [], start
    while i < len(messages):
        j = i + 1
        if messages[i].get("tool_calls"):
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
        out.append((i, j))
        i = j
    return out


def _tokens(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or "")) for m in messages) // 3


def trim(messages: list[dict[str, Any]], ctx: int, *, high: float = 0.85, low: float = 0.5) -> bool:
    """When the conversation passes ``high`` of ``ctx``, drop whole oldest turns (after the system
    message and the task) down to ``low`` and leave one fixed marker; true when it did. Nothing kept
    is edited, so from then on the prompt only grows and the server's cached prefix is reused."""
    if _tokens(messages) <= ctx * high:
        return False
    head, rest = messages[:2], messages[2:]
    if rest and rest[0].get("content") == MARKER:
        rest = rest[1:]
    dropped = 0
    for _, end in _groups(rest, 0)[:-1]:
        if _tokens(head + rest[dropped:]) <= ctx * low:
            break
        dropped = end
    if not dropped:
        return False
    messages[:] = [*head, {"role": "user", "content": MARKER}, *rest[dropped:]]
    return True
