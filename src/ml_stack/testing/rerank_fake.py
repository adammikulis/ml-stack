"""What the fake llama-server answers on ``/v1/rerank``."""

from __future__ import annotations

from typing import Any


def rerank_results(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Each document scores the share of the query's words it contains, listed in document
    order (the real server does not sort either) and cut to the ``top_n`` best when asked."""
    words = set(str(body.get("query") or "").lower().split())
    docs = [str(one).lower().split() for one in body.get("documents") or []]
    rows = [{"index": i, "relevance_score": len(words & set(doc)) / max(1, len(words))}
            for i, doc in enumerate(docs)]
    top = body.get("top_n")
    if isinstance(top, int) and top > 0:
        keep = {r["index"] for r in sorted(rows, key=lambda r: -r["relevance_score"])[:top]}
        rows = [r for r in rows if r["index"] in keep]
    return rows
