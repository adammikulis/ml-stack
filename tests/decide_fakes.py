"""Stand-ins for the models behind poolhouse.decide: a chat server that reports log-probabilities
and an embedder that hashes words."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable

Scorer = Callable[[str], dict[str, float]]
"""``scorer(user message) -> {letter: probability}``"""


def logprob_handler(scorer: Scorer, *, seen: list[dict] | None = None):
    """A handler for the ``server`` fixture that answers /v1/chat/completions the way a
    llama-server does: ``top_logprobs`` for the first generated token."""
    def handle(method: str, path: str, body: bytes) -> tuple[int, bytes]:
        if path == "/v1/models":
            return 200, json.dumps({"data": [{"id": "qwen3-fake"}]}).encode()
        sent = json.loads(body)
        if seen is not None:
            seen.append(sent)
        user = next(m["content"] for m in reversed(sent["messages"]) if m["role"] == "user")
        probs = scorer(user)
        top = [{"token": letter, "logprob": math.log(p)} for letter, p in probs.items() if p > 0]
        top.append({"token": " the", "logprob": -12.0})
        return 200, json.dumps({"model": "qwen3-fake", "choices": [{
            "message": {"content": top[0]["token"]},
            "logprobs": {"content": [{"token": top[0]["token"], "top_logprobs": top}]}}]}).encode()
    return handle


def hashed_embedding(text: str, dim: int = 64) -> list[float]:
    """A bag-of-words vector: each word adds a fixed pseudo-random direction."""
    out = [0.0] * dim
    for word in re.findall(r"[a-z0-9_]+", text.lower()):
        digest = hashlib.sha256(word.encode()).digest()
        for i in range(dim):
            out[i] += (digest[i % 32] / 255.0 - 0.5) * (1.0 if i < 32 else -1.0)
    return out


def embedder(texts: list[str]) -> list[list[float]]:
    """The embedder `EmbedDecider` takes, over `hashed_embedding`."""
    return [hashed_embedding(t) for t in texts]
