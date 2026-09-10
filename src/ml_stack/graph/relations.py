"""The verbs a fact is kept under, the relations that must form a DAG, and the rings among
them."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ml_stack.graph.dag import topological_order

__all__ = ["HIERARCHY", "INVERSES", "canonical_direction", "cycles", "rings"]

# The direction a fact is kept in, and the verbs that say the same thing the other way.
INVERSES: dict[str, frozenset[str]] = {
    "part_of": frozenset({"has_part", "contains", "has"}),
    "causes": frozenset({"caused_by"}),
    "precedes": frozenset({"follows", "after"}),
    "produces": frozenset({"produced_by", "made_by"}),
    "created_by": frozenset({"authored", "wrote", "created", "built", "proposed"}),
    "requires": frozenset({"required_by", "enables"}),
    "supersedes": frozenset({"superseded_by", "replaced_by"}),
}
"""``{canonical verb: the verbs that state it with the ends swapped}``. ``X has_part Y`` is
``Y part_of X``; the pass keeps the left-hand form."""

#: The relations that say one thing is under another, each of which must be a DAG: nothing
#: is part of itself at one remove, and nobody reports to somebody who reports to them.
HIERARCHY: tuple[str, ...] = ("part_of", "reports_to", "contains", "member_of", "supersedes")


def canonical_direction(rel: str) -> tuple[str, bool]:
    """``(canonical verb, flipped)``: the verb a fact is kept under, and whether the ends
    must be swapped to get there."""
    if rel in INVERSES:
        return rel, False
    for keep, others in INVERSES.items():
        if rel in others:
            return keep, True
    return rel, False


def cycles(edges: Iterable[Mapping[str, Any]], *,
           relations: Iterable[str] = HIERARCHY) -> list[tuple[str, list[str]]]:
    """Every cycle among the hierarchical relations: ``(relation, the ids round it)``.

    Each relation is taken on its own, since ``part_of`` running in a ring is a fault
    whether or not ``reports_to`` does.
    """
    wanted = list(dict.fromkeys(str(r) for r in relations))
    by_rel: dict[str, list[tuple[str, str]]] = {}
    for edge in edges:
        rel = str(edge.get("rel") or "")
        if rel in wanted:
            by_rel.setdefault(rel, []).append((str(edge.get("source")), str(edge.get("target"))))

    found: list[tuple[str, list[str]]] = []
    for rel in wanted:
        pairs = by_rel.get(rel) or []
        ids: list[str] = []
        at: dict[str, int] = {}
        for ends in pairs:
            for node_id in ends:
                if node_id not in at:
                    at[node_id] = len(ids)
                    ids.append(node_id)
        src = [at[a] for a, _ in pairs]
        dst = [at[b] for _, b in pairs]
        if not ids or topological_order(len(ids), src, dst) is not None:
            continue
        found += [(rel, [ids[i] for i in ring]) for ring in rings(len(ids), src, dst)]
    return found



def rings(n: int, src: list[int], dst: list[int]) -> list[list[int]]:
    """One cycle per tangle, as node indices in the order the edges run round it."""
    from collections import deque

    outgoing: list[list[int]] = [[] for _ in range(n)]
    incoming: list[list[int]] = [[] for _ in range(n)]
    for u, v in zip(src, dst):
        outgoing[u].append(v)
        incoming[v].append(u)

    left = [len(row) for row in outgoing]
    alive = [True] * n
    queue = deque(u for u in range(n) if left[u] == 0)
    while queue:
        u = queue.popleft()
        alive[u] = False
        for p in incoming[u]:
            left[p] -= 1
            if left[p] == 0:
                queue.append(p)

    rings: list[list[int]] = []
    already: set[frozenset[int]] = set()
    walked = [False] * n
    for start in range(n):
        if not alive[start] or walked[start]:
            continue
        walk: list[int] = []
        seen: dict[int, int] = {}
        u = start
        while u not in seen:
            seen[u] = len(walk)
            walk.append(u)
            walked[u] = True
            u = next(v for v in outgoing[u] if alive[v])
        ring = walk[seen[u]:]
        if frozenset(ring) not in already:
            already.add(frozenset(ring))
            rings.append(ring)
    return rings
