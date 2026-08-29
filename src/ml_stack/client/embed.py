"""Embeddings from a local server, and the two similarity helpers everyone rewrites."""

from __future__ import annotations

import math
from typing import Any

from ml_stack.client.http import ServerError, request_json


class EmbeddingError(ServerError):
    """The embedding request failed, or came back the wrong shape."""


def embed(
    texts: str | list[str],
    *,
    base_url: str = "http://127.0.0.1:8080",
    model: str | None = None,
    expect_dim: int | None = None,
    timeout: float = 60.0,
    tries: int = 3,
    api_key: str | None = None,
) -> list[list[float]]:
    """Embed one string or a batch. Always returns a list of vectors."""
    batch = [texts] if isinstance(texts, str) else list(texts)
    if not batch:
        return []

    body: dict[str, Any] = {"input": batch}
    if model:
        body["model"] = model

    payload = request_json(
        f"{base_url.rstrip('/')}/v1/embeddings",
        payload=body,
        timeout=timeout,
        tries=tries,
        headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
    )

    if not isinstance(payload, dict) or "data" not in payload:
        raise EmbeddingError(f"embedding response has no 'data': {str(payload)[:200]}")

    vectors: list[list[float]] = []
    for entry in payload["data"]:
        raw = entry.get("embedding") if isinstance(entry, dict) else None
        if not isinstance(raw, list):
            raise EmbeddingError(f"embedding entry has no vector: {str(entry)[:200]}")
        # llama-server returns [[...]] for pooled embeddings on some builds.
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        vectors.append([float(v) for v in raw])

    if len(vectors) != len(batch):
        raise EmbeddingError(f"asked for {len(batch)} embeddings, got {len(vectors)}")

    if expect_dim is not None:
        for i, vector in enumerate(vectors):
            if len(vector) != expect_dim:
                raise EmbeddingError(
                    f"embedding {i} has dimension {len(vector)}, expected {expect_dim}. "
                    "The server is serving a different model than this index was built "
                    "with -- padding it to fit would corrupt the index silently."
                )

    return vectors


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Raises on a dimension mismatch rather than truncating."""
    if len(a) != len(b):
        raise EmbeddingError(f"cannot compare vectors of length {len(a)} and {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if norm == 0.0 else dot / norm


def top_k(
    query: list[float],
    candidates: list[list[float]],
    k: int = 10,
) -> list[tuple[int, float]]:
    """The ``k`` most similar candidates as ``(index, score)``, best first."""
    scored = [(i, cosine(query, c)) for i, c in enumerate(candidates)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def rank_pairs(
    vectors: list[list[float]],
    *,
    limit: int | None = None,
) -> list[tuple[int, int, float]]:
    """Every distinct pair as ``(i, j, score)`` with ``i < j``, most similar first."""
    scored = [
        (i, j, cosine(vectors[i], vectors[j]))
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    scored.sort(key=lambda triple: (-triple[2], triple[0], triple[1]))
    return scored[:limit] if limit is not None else scored
