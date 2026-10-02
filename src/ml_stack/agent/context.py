"""How much of a model's context a conversation fills.

Tokens are counted by the server's own ``/tokenize`` when the client can reach it. When it
cannot, the count is the heuristic of `ml_stack.client.tokens`, rescaled by the
characters-per-token ratio seen in whatever the server did count, and padded by a margin.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ml_stack.client.health import serving_params
from ml_stack.client.tokens import estimate_tokens
from ml_stack.http import ServerError

__all__ = [
    "ContextUsage",
    "Counter",
    "context_limit",
    "context_usage",
    "message_text",
    "message_tokens",
]

MARGIN = 1.15
"""Applied to an estimate no server has checked."""

PER_MESSAGE = 4
"""Tokens a chat template spends on a message's role markers."""


@dataclass(frozen=True, slots=True)
class ContextUsage:
    """Tokens used out of ``limit``; ``exact`` is True when the server did the counting."""

    used: int
    limit: int
    fraction: float
    exact: bool

    def as_dict(self) -> dict[str, Any]:
        return {"used": self.used, "limit": self.limit, "fraction": self.fraction,
                "exact": self.exact}


class Counter:
    """Counts tokens in text, remembering each answer."""

    def __init__(self, client: Any = None, *, margin: float = MARGIN) -> None:
        self.client = client
        self.margin = margin
        self.exact = hasattr(client, "tokenize")
        self._seen: dict[bytes, int] = {}
        self._chars = 0
        self._tokens = 0

    def __call__(self, text: str) -> int:
        if not text:
            return 0
        key = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=12).digest()
        if key not in self._seen:
            self._seen[key] = self._count(text)
        return self._seen[key]

    def _count(self, text: str) -> int:
        if self.exact:
            try:
                n = len(self.client.tokenize(text))
            except ServerError:
                self.exact = False
            else:
                self._chars += len(text)
                self._tokens += n
                return n
        if self._tokens:
            return math.ceil(len(text) * self._tokens / self._chars * 1.05) + 1
        return math.ceil(estimate_tokens(text) * self.margin)


def message_text(message: Mapping[str, Any]) -> str:
    """The text of a message as a template would render it: content, calls and name."""
    content = message.get("content")
    if isinstance(content, list):
        content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, Mapping))
    parts = [str(content or "")]
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        args = fn.get("arguments")
        parts.append(f"{fn.get('name', '')} " + (args if isinstance(args, str)
                                                  else json.dumps(args, sort_keys=True)))
    return "\n".join(p for p in parts if p)


def message_tokens(message: Mapping[str, Any], count: Callable[[str], int]) -> int:
    """Tokens one message takes, role markers included."""
    return count(message_text(message)) + PER_MESSAGE


def context_usage(messages: Sequence[Mapping[str, Any]],
                  tools: Sequence[Mapping[str, Any]] | None, context_size: int, *,
                  count: Callable[[str], int] | None = None) -> ContextUsage:
    """What ``messages`` and the offered ``tools`` take of a ``context_size``-token window."""
    count = count or Counter()
    used = sum(message_tokens(m, count) for m in messages)
    if tools:
        used += count(json.dumps(tools, sort_keys=True))
    return ContextUsage(used, context_size, round(used / max(1, context_size), 4),
                        bool(getattr(count, "exact", False)))


def context_limit(client: Any) -> int | None:
    """The per-slot context the server behind ``client`` reports, or None."""
    params = serving_params(client.base_url)
    return params.n_ctx if params else None
