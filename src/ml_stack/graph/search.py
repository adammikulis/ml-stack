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
# How many of the fused hits are re-ordered by meaning before they are handed over. Fusion
# decides which entries come back; this decides which of them a model reads first, and a
# model reads the first few properly and skims the rest. Six is about a screenful -- past
# it the order stopped mattering, because nothing looked.
RERANK = 6

# What a lexical rank means, for a hit to say why it matched. The whole label and part of it
# are both "label" -- what tells them apart is the rank, and the score carries that. The
# names are the voters a rich `hybrid` hit lists under "matched", beside the store's two.
MATCHED_BY_RANK = {4: "label", 3: "label", 2: "attribute", 1: "said"}
WORDS = "words"
MEANING = "meaning"


def rrf_scored(*rankings: Sequence[str], k: int = RRF_K,
               limit: int = LIMIT) -> list[tuple[str, float]]:
    """Fused ids with the score that placed them, best first.

    For a hit that says how well it did: one vote at the top is worth 1/(k+1), and the
    number only means something against the others in the same list.
    """
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, node_id in enumerate(ranking):
            score[node_id] = score.get(node_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def rrf(*rankings: Sequence[str], k: int = RRF_K, limit: int = LIMIT) -> list[str]:
    """Fuse ranked lists of ids. Appearing well in two beats appearing first in one."""
    return [node_id for node_id, _ in rrf_scored(*rankings, k=k, limit=limit)]


def lexical(graph: Mapping[str, Any], text: str, *, limit: int = LIMIT,
            rich: bool = False) -> list[Any]:
    """Ids whose label, attributes or own words carry the characters, best first.

    Ranked: the whole label, then part of it, then an attribute, then something that was said.
    With ``rich``, each comes as ``{"id", "score", "matched"}`` -- the rank it was found at
    and the one thing that found it, named as ``hybrid`` names its voters -- so a caller can
    tell an exact label from one word in one quote.
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
    if not rich:
        return [node_id for _, _, node_id in scored[:limit]]
    return [{"id": node_id, "score": rank, "matched": [MATCHED_BY_RANK[rank]]}
            for rank, _, node_id in scored[:limit]]


def reranked(rows: Sequence[Mapping[str, Any]], near: Mapping[str, float], *,
             top: int = RERANK) -> list[dict[str, Any]]:
    """Those hits with the first ``top`` put in order of how close each is to the question.

    Fusion answers "which entries", by agreement between three ways of looking; it cannot
    answer "which of these did the asker mean", because a reciprocal rank is a vote count
    and not a distance. The vectors can, and they are already in the store -- so once the
    field is narrow enough to be worth reading, the hits the embedder actually knows are
    put in the order it puts them.

    Membership never changes and nothing outside the window moves, which is what makes this
    cheap to measure: the same entries come back, read in a different order. A hit the
    vectors have never seen keeps its place rather than sinking, because an exact label
    match with no embedding is still the right answer and must not be pushed down the page
    by a candidate the embedder merely likes. Equal scores keep the order fusion gave them.
    """
    window = list(rows[:top])
    spots = [i for i, row in enumerate(window) if str(row.get("id") or "") in near]
    if len(spots) < 2:
        return list(rows)
    moved = sorted((window[i] for i in spots), key=lambda row: -near[str(row["id"])])
    for spot, row in zip(spots, moved, strict=True):
        window[spot] = row
    return window + list(rows[top:])


def hybrid(graph: Mapping[str, Any], text: str, *, store: Any = None,
           vector: Sequence[float] | None = None, model: str = "",
           limit: int = LIMIT, rich: bool = False,
           rerank: bool = True) -> list[dict[str, Any]]:
    """The three ways, fused. Whatever is unavailable simply does not vote.

    ``store`` supplies the word index and the vectors; ``vector`` is the question already
    embedded, which the caller does because only it knows which embedder the graph was built
    with.

    Each hit is ``{"id", "label", "kind"}``. With ``rich`` it also says how it got there:
    ``"score"`` is the fused score to three places, and ``"matched"`` names the voters that
    found it -- ``"label"``, ``"attribute"`` or ``"said"`` from the characters, ``"words"``
    from the word index, ``"meaning"`` from the vectors -- so a reader can tell an exact
    label from one word in one quote. Off, the hit is exactly what it always was.

    ``rerank`` puts the first `RERANK` fused hits in order of meaning -- see `reranked` --
    whenever the store holds vectors for them and the question arrived embedded. It is on
    because it changes the order and not the membership, which is the cheapest kind of
    change to be wrong about: the same entries come back either way, and what moves is
    which of them a model reads first. ``rerank=False`` is the fused order as it was.
    """
    found = lexical(graph, text, limit=limit * 2, rich=True)
    rankings: list[list[str]] = [[r["id"] for r in found]]
    voters: dict[str, list[str]] = {r["id"]: list(r["matched"]) for r in found}
    near: dict[str, float] = {}

    def vote(ranking: list[str], name: str) -> None:
        rankings.append(ranking)
        for node_id in ranking:
            mine = voters.setdefault(node_id, [])
            if name not in mine:
                mine.append(name)

    if store is not None:
        try:
            vote([r["id"] for r in store.search(text, limit=limit * 2)], WORDS)
        except Exception:  # noqa: BLE001 - no index yet is not an error, it is one fewer vote
            pass
        if vector is not None:
            try:
                # More than the votes are wanted when reranking: a hit fused out of the
                # characters alone is only re-ordered if the vectors have an opinion on it,
                # so the pool asked for is wider than the pool that votes. What votes is
                # the same `limit * 2` either way -- what fusion returns must not depend on
                # whether the order is about to be adjusted.
                close = list(store.similar(vector, model=model,
                                           limit=limit * (4 if rerank else 2)))
                vote([r["id"] for r in close][:limit * 2], MEANING)
                near = {str(r["id"]): float(r["similarity"]) for r in close
                        if isinstance(r.get("similarity"), (int, float))}
            except Exception:  # noqa: BLE001
                pass
    known = {str(n["id"]): n for n in (graph.get("nodes") or ())}
    rows: list[dict[str, Any]] = []
    for node_id, score in rrf_scored(*rankings, limit=limit):
        if node_id not in known:
            continue
        row: dict[str, Any] = {"id": node_id, "label": str(known[node_id].get("label") or ""),
                               "kind": str(known[node_id].get("kind") or "")}
        if rich:
            row["score"] = round(score, 3)
            row["matched"] = list(voters.get(node_id, ()))
        rows.append(row)
    return reranked(rows, near) if rerank and near else rows
