"""A reranker against the vectors and the fused order, on retrieval alone.

`hybrid` is run three ways over every scored question in `ml_stack.graph.community.QUESTIONS`
(fused, the store's vector order, and a served model reranker) and `table` prints recall, MRR
and nDCG for each; questions expecting nobody are counted apart. It ranks by characters alone
without a store, and touches the network only for a URL it is given:
``ml-stack-bench retrieval --store S --embed-url U --rerank-url R``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from ml_stack.bench.ranking import ndcg_at, recall_at, reciprocal_rank
from ml_stack.client.embed import QUERY, embed
from ml_stack.client.rerank import rerank
from ml_stack.graph.community import QUESTIONS, graph as invented
from ml_stack.graph.search import RERANK, hybrid
from ml_stack.graph.store import GraphStore

MODES = ("fused", "vectors", "reranker")
Reranker = Callable[[str, list[dict[str, Any]]], list[float]]


def node_texts(graph: Mapping[str, Any]) -> dict[str, str]:
    """What a reranker reads for each node: its label and everything it said."""
    said = graph.get("messages") or {}
    return {str(n["id"]): " ".join(
        [str(n.get("label") or ""), *(str((said.get(m) or {}).get("text") or "")
                                      for m in n.get("messages") or ())]).strip()
            for n in graph.get("nodes") or ()}


def model_reranker(graph: Mapping[str, Any], url: str, model: str = "") -> Reranker:
    """A `hybrid` reranker that asks the server at ``url`` to score each hit's text."""
    texts = node_texts(graph)
    return lambda question, rows: rerank(
        question, [texts.get(str(r["id"]), str(r.get("label") or "")) for r in rows],
        base_url=url, model=model or None)


@dataclass(frozen=True)
class Finder:
    """What a search is run against: the graph, its store, how a question is embedded as the
    store's vectors were, the model they are filed under and the reranker, if any."""

    graph: Mapping[str, Any]
    store: Any = None
    embed: Callable[[str], list[float] | None] = lambda text: None
    model: str = ""
    reranker: Reranker | None = None

    def ranked(self, question: str, mode: str) -> list[str]:
        """The ids `hybrid` returns for ``question`` in one of the three `MODES`."""
        if mode not in MODES:
            raise ValueError(f"mode is one of {', '.join(MODES)}, not {mode!r}")
        if mode == "reranker" and self.reranker is None:
            raise ValueError("the reranker mode needs a reranker")
        how: Any = self.reranker if mode == "reranker" else mode == "vectors"
        return [r["id"] for r in hybrid(self.graph, question, store=self.store,
                                        vector=self.embed(question), model=self.model,
                                        rerank=how)]


def score(lists: Sequence[tuple[list[str], Sequence[str]]], k: int = RERANK) -> dict[str, float]:
    """Mean recall@k, MRR and nDCG@k over ``(ranked ids, expected ids)`` pairs."""
    n = max(1, len(lists))
    return {f"recall@{k}": sum(recall_at(r, e, k) for r, e in lists) / n,
            "mrr": sum(reciprocal_rank(r, e) for r, e in lists) / n,
            f"ndcg@{k}": sum(ndcg_at(r, e, k) for r, e in lists) / n}


def compare(questions: Sequence[Mapping[str, Any]], *, finder: Finder,
            modes: Sequence[str] = MODES, k: int = RERANK) -> dict[str, dict[str, float]]:
    """Each mode's scores over the questions that expect somebody."""
    scored = [q for q in questions if q.get("expect")]
    return {mode: score([(finder.ranked(str(q["q"]), mode), list(q["expect"]))
                         for q in scored], k)
            for mode in modes}


def table(results: Mapping[str, Mapping[str, float]], *, asked: int, nobody: int) -> str:
    """The modes as rows, the metrics as columns, and what was left out."""
    columns = list(next(iter(results.values()))) if results else []
    lines = [f"{'mode':<10}" + "".join(f"{c:>10}" for c in columns)]
    lines += [f"{mode:<10}" + "".join(f"{row[c]:>10.3f}" for c in columns)
              for mode, row in results.items()]
    return "\n".join([*lines, f"{asked} questions scored; {nobody} expect nobody and are "
                      "not in these columns"])


def embedder(url: str, model: str) -> Callable[[str], list[float] | None]:
    """The question embedded as the store's vectors were, or nothing when there is no server."""
    if not url:
        return lambda text: None
    return lambda text: embed([QUERY + text], base_url=url, model=model or None)[0]


def report(args: Any) -> str:
    """The comparison table for the options ``ml-stack-bench retrieval`` was given."""
    graph = invented()
    url = str(args.rerank_url or "")
    reranker = model_reranker(graph, url, str(args.rerank_model)) if url else None
    opened: Any = GraphStore(args.store, read_only=True) if args.store else nullcontext(None)
    with opened as held:
        results = compare(QUESTIONS, modes=MODES if reranker else MODES[:2], k=int(args.k),
                          finder=Finder(graph, held, embedder(args.embed_url, args.embed_model),
                                        str(args.embed_model), reranker))
    scored = sum(1 for q in QUESTIONS if q.get("expect"))
    return table(results, asked=scored, nobody=len(QUESTIONS) - scored)
