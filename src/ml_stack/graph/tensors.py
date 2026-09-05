"""The mapping graph -- ``{"nodes": [...], "edges": [...]}`` -- as a tensor `Graph`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ml_stack.graph.data import Graph

__all__ = ["relations_in", "tensors"]


def relations_in(graph: Mapping[str, Any]) -> list[str]:
    """Every relation the edges use, in the order they first appear."""
    seen: list[str] = []
    for edge in graph.get("edges") or ():
        rel = str(edge.get("rel") or "")
        if rel not in seen:
            seen.append(rel)
    return seen


def tensors(graph: Mapping[str, Any], *, backend: Any = None,
            relations: Sequence[str] | None = None) -> tuple[Graph, list[str]]:
    """The graph as arrays, and its node ids in index order.

    ``edge_type`` indexes the relation vocabulary, which is ``relations`` when given and the
    graph's own otherwise; the vocabulary is on the result's ``meta["relations"]``. ``w`` is
    each edge's ``weight``, 1.0 without one. An edge naming a node the graph does not hold,
    or a relation outside a given vocabulary, is left out.
    """
    from ml_stack.train.backend import get_backend

    ops = (backend or get_backend()).ops
    ids = [str(n.get("id")) for n in graph.get("nodes") or () if n.get("id") is not None]
    at = {node_id: i for i, node_id in enumerate(ids)}
    vocabulary = list(relations) if relations is not None else relations_in(graph)
    kinds = {rel: i for i, rel in enumerate(vocabulary)}

    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []
    types: list[int] = []
    for edge in graph.get("edges") or ():
        source, target = at.get(str(edge.get("source"))), at.get(str(edge.get("target")))
        kind = kinds.get(str(edge.get("rel") or ""))
        if source is None or target is None or kind is None:
            continue
        src.append(source)
        dst.append(target)
        weights.append(float(edge.get("weight", 1.0) or 0.0))
        types.append(kind)

    made = Graph(
        num_nodes=len(ids),
        src=ops.array(src, dtype=ops.int32),
        dst=ops.array(dst, dtype=ops.int32),
        w=ops.array(weights, dtype=ops.float32),
        edge_type=ops.array(types, dtype=ops.int32),
        meta={"relations": vocabulary},
    )
    return made, ids
