"""Relevance scores from a local reranking server (llama-server --reranking)."""

from __future__ import annotations

from typing import Any

from ml_stack.http import ServerError, request_json


class RerankError(ServerError):
    """The rerank request failed, or came back the wrong shape."""


def rerank(query: str, documents: list[str], *, base_url: str = "http://127.0.0.1:8080",
           model: str | None = None, top_n: int | None = None) -> list[float]:
    """One relevance score per document, in the order the documents were given.

    The server answers ``results[{index, relevance_score}]`` in whatever order it likes;
    the scores are put back by ``index``. With ``top_n`` the server may leave documents
    out, and those score ``float("-inf")`` so they sort last.
    """
    if not documents:
        return []
    body: dict[str, Any] = {"query": query, "documents": list(documents)}
    if model:
        body["model"] = model
    if top_n is not None:
        body["top_n"] = top_n
    payload = request_json(f"{base_url.rstrip('/')}/v1/rerank", payload=body, timeout=60.0)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RerankError(f"rerank response has no 'results': {str(payload)[:200]}")
    scores = [float("-inf")] * len(documents)
    for entry in payload["results"]:
        index = entry.get("index") if isinstance(entry, dict) else None
        score = entry.get("relevance_score") if isinstance(entry, dict) else None
        if not isinstance(index, int) or not 0 <= index < len(documents) or score is None:
            raise RerankError(f"rerank entry is malformed: {str(entry)[:200]}")
        scores[index] = float(score)
    return scores
