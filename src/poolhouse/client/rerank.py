"""Relevance scores from a local reranking server (llama-server --reranking)."""

from __future__ import annotations

import json
import math
import time
from typing import Any

from poolhouse.http import ServerError, open_stream, shown


class RerankError(ServerError):
    """The rerank request failed, or came back the wrong shape."""


MAX_REPLY = 4 * 1024 * 1024
TIMEOUT = 60.0  # seconds, per socket wait and for the whole answer


def _read(url: str, body: dict[str, Any], timeout: float) -> bytes:
    """The reply body, read in pieces under one deadline and refused past ``MAX_REPLY``.

    ``timeout`` bounds each socket wait and, as a deadline, the whole read, so a server that
    drips bytes cannot hold the caller past it.
    """
    deadline = time.monotonic() + timeout
    data = json.dumps(body).encode("utf-8")
    try:
        with open_stream(url, data=data, method="POST", timeout=timeout,
                         headers={"Content-Type": "application/json"}) as response:
            raw = b""
            while len(raw) <= MAX_REPLY:
                if time.monotonic() > deadline:
                    raise RerankError(f"{shown(url)} did not finish answering in {timeout:g}s")
                piece = response.read1(65536)
                if not piece:
                    return response.sealed.open(int(response.status), response.headers, raw)
                raw += piece
    except RerankError:
        raise
    except ServerError as exc:
        raise RerankError(str(exc), status=exc.status, body=exc.body) from exc
    except (TimeoutError, OSError, ValueError) as exc:
        raise RerankError(f"{shown(url)} failed while answering: {exc}") from exc
    raise RerankError(f"{shown(url)} answered more than {MAX_REPLY} bytes")


def _score(entry: Any, size: int) -> tuple[int, float]:
    """``(index, score)`` of one result row, or a RerankError. A bool is not an index and a
    score must be a finite number (NaN and infinity are refused, not clamped)."""
    index = entry.get("index") if isinstance(entry, dict) else None
    score = entry.get("relevance_score") if isinstance(entry, dict) else None
    if (not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < size
            or not isinstance(score, (int, float)) or isinstance(score, bool)
            or not math.isfinite(score)):
        raise RerankError(f"rerank entry is malformed: {str(entry)[:200]}")
    return index, float(score)


def rerank(query: str, documents: list[str], *, base_url: str = "http://127.0.0.1:8080",
           model: str | None = None, top_n: int | None = None) -> list[float]:
    """One relevance score per document, in the order the documents were given.

    The server answers ``results[{index, relevance_score}]`` in whatever order it likes;
    the scores are put back by ``index``. With ``top_n`` the server may leave documents
    out, and those score ``float("-inf")`` so they sort last. Every other reply, however
    hostile (too big, too slow, not JSON, the wrong shape, an index or score that is not a
    number in range) raises RerankError.
    """
    if not documents:
        return []
    body: dict[str, Any] = {"query": query, "documents": list(documents)}
    if model:
        body["model"] = model
    if top_n is not None:
        body["top_n"] = top_n
    url = f"{base_url.rstrip('/')}/v1/rerank"
    try:
        payload = json.loads(_read(url, body, TIMEOUT).decode("utf-8") or "null")
    except (ValueError, RecursionError) as exc:
        raise RerankError(f"{shown(url)} returned non-JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RerankError(f"rerank response has no 'results': {str(payload)[:200]}")
    scores = [float("-inf")] * len(documents)
    for entry in payload["results"]:
        index, score = _score(entry, len(documents))
        scores[index] = score
    return scores
