"""Finding things in a graph three ways at once.

Each way is good at something the others are bad at. Matching the characters finds a name
somebody typed exactly, and finds nothing when they typed it differently. A word index stems
and ranks, so "compiler" finds "compilers", and it still needs the word. Vectors find what a
thing means, so "who fixes machines" finds a robotics technician, and they will confidently
return something loosely related when the exact answer was there all along.

So all three run and the rankings are fused. Reciprocal rank fusion does that without anyone
having to pretend a cosine similarity and a BM25 score are the same kind of number: a result
is worth 1/(k + its rank) in each list it appears in, and appearing well in two lists beats
appearing first in one.

    hits = hybrid(graph, "robotics", store=store, vector=embed("robotics"))
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

RRF_K = 60
LIMIT = 12


def rrf(*rankings: Sequence[str], k: int = RRF_K, limit: int = LIMIT) -> list[str]:
    """Fuse ranked lists of ids. Appearing well in two beats appearing first in one."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, node_id in enumerate(ranking):
            score[node_id] = score.get(node_id, 0.0) + 1.0 / (k + rank + 1)
    return [node_id for node_id, _ in
            sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]


def lexical(graph: Mapping[str, Any], text: str, *, limit: int = LIMIT) -> list[str]:
    """Ids whose label, attributes or own words carry the characters, best first.

    Ranked: the whole label, then part of it, then an attribute, then something that was said.
    """
    want = " ".join((text or "").split()).casefold()
    if not want:
        return []
    messages = graph.get("messages") or {}
    scored: list[tuple[int, int, str]] = []
    for node in graph.get("nodes") or ():
        label = str(node.get("label") or "").casefold()
        attrs = node.get("attrs") or {}
        if label == want:
            rank = 4
        elif want in label:
            rank = 3
        elif any(want in str(v).casefold() for v in attrs.values()):
            rank = 2
        elif any(want in str((messages.get(mid) or {}).get("text") or "").casefold()
                 for mid in (node.get("messages") or ())[:20]):
            rank = 1
        else:
            continue
        scored.append((rank, int(node.get("mentions") or 0), str(node["id"])))
    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [node_id for _, _, node_id in scored[:limit]]


def hybrid(graph: Mapping[str, Any], text: str, *, store: Any = None,
           vector: Sequence[float] | None = None, model: str = "",
           limit: int = LIMIT) -> list[dict[str, str]]:
    """The three ways, fused. Whatever is unavailable simply does not vote.

    ``store`` supplies the word index and the vectors; ``vector`` is the question already
    embedded, which the caller does because only it knows which embedder the graph was built
    with.
    """
    rankings: list[list[str]] = [lexical(graph, text, limit=limit * 2)]
    if store is not None:
        try:
            rankings.append([r["id"] for r in store.search(text, limit=limit * 2)])
        except Exception:  # noqa: BLE001 - no index yet is not an error, it is one fewer vote
            pass
        if vector is not None:
            try:
                rankings.append([r["id"] for r in
                                 store.similar(vector, model=model, limit=limit * 2)])
            except Exception:  # noqa: BLE001
                pass
    known = {str(n["id"]): n for n in (graph.get("nodes") or ())}
    return [{"id": i, "label": str(known[i].get("label") or ""),
             "kind": str(known[i].get("kind") or "")}
            for i in rrf(*rankings, limit=limit) if i in known]
