"""What a `supersedes` edge says about the rest of a graph: which nodes trace back to a
source something has replaced, and what still touches them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SUPERSEDES = "supersedes"
PROVENANCE = "read_from"


def superseded(graph: Mapping[str, Any]) -> dict[str, str]:
    """Node id -> the label of what replaced its source.

    Walks every ``supersedes`` edge, marks its target superseded, and follows every
    :data:`PROVENANCE` edge into that target so a node read from a replaced source is
    marked too.
    """
    nodes = list(graph.get("nodes") or ())
    edges = list(graph.get("edges") or ())
    label = {str(n["id"]): str(n.get("label", "")) for n in nodes}

    replaced_by: dict[str, str] = {}
    for edge in edges:
        if edge.get("rel") == SUPERSEDES:
            old, new = str(edge.get("target")), str(edge.get("source"))
            replaced_by[old] = label.get(new, new)

    out = dict(replaced_by)
    for edge in edges:
        if edge.get("rel") != PROVENANCE:
            continue
        node_id, source_id = str(edge.get("source")), str(edge.get("target"))
        if source_id in replaced_by:
            out[node_id] = replaced_by[source_id]
    return out


def resting_on(graph: Mapping[str, Any], node_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Every node whose edge points at one of ``node_ids``, and the edge that does it."""
    wanted = {str(i) for i in node_ids}
    label = {str(n["id"]): str(n.get("label", "")) for n in (graph.get("nodes") or ())}
    out: list[dict[str, Any]] = []
    for edge in graph.get("edges") or ():
        source, target = str(edge.get("source")), str(edge.get("target"))
        if target in wanted and source not in wanted:
            out.append({"id": source, "label": label.get(source, source),
                        "rel": edge.get("rel"), "on": target})
    return out
