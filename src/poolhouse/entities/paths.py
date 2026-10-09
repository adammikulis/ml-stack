"""Getting from one node to another.

A question about two people is usually a question about what stands between them. The graph
already holds the answer; what is needed is the cheapest way across it. Weight is how many
messages agree on a link, so a well-attested link is a short step and a link seen once is a
long one — the cheapest path is the best-evidenced one, not merely the shortest.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping
from typing import Any

Edge = Mapping[str, Any]


def adjacency(edges: Iterable[Edge], *, weight_key: str = "weight") -> dict[str, list[tuple[str, float, Edge]]]:
    """node id -> [(other id, cost, the edge)], both ways along every edge."""
    out: dict[str, list[tuple[str, float, Edge]]] = {}
    for e in edges:
        a, b = e.get("source"), e.get("target")
        if not a or not b or a == b:
            continue
        cost = 1.0 / max(1.0, float(e.get(weight_key) or 1))
        out.setdefault(a, []).append((b, cost, e))
        out.setdefault(b, []).append((a, cost, e))
    return out


def shortest_path(edges: Iterable[Edge], start: str, goal: str, *,
                  weight_key: str = "weight", hops: int = 8) -> list[Edge]:
    """The best-evidenced way from ``start`` to ``goal``, as the edges to walk.

    Empty when there is no way across within ``hops``. Dijkstra rather than A*: the graph has no
    geometry to guess distance from, so any admissible heuristic here is zero, and A* with a zero
    heuristic is Dijkstra with extra words.
    """
    if start == goal:
        return []
    near = adjacency(edges, weight_key=weight_key)
    if start not in near or goal not in near:
        return []
    best: dict[str, float] = {start: 0.0}
    came: dict[str, tuple[str, Edge]] = {}
    queue: list[tuple[float, int, str]] = [(0.0, 0, start)]
    seen: set[str] = set()
    while queue:
        cost, depth, here = heapq.heappop(queue)
        if here in seen:
            continue
        seen.add(here)
        if here == goal:
            break
        if depth >= hops:
            continue
        for other, step, edge in near.get(here, ()):
            if other in seen:
                continue
            through = cost + step
            if through < best.get(other, float("inf")):
                best[other] = through
                came[other] = (here, edge)
                heapq.heappush(queue, (through, depth + 1, other))
    if goal not in came:
        return []
    walk: list[Edge] = []
    here = goal
    while here != start:
        here, edge = came[here]
        walk.append(edge)
    walk.reverse()
    return walk


def between(edges: Iterable[Edge], start: str, goal: str, **kw: Any) -> list[str]:
    """The node ids along the path, ends included. Empty when there is no way across."""
    walk = shortest_path(edges, start, goal, **kw)
    if not walk:
        return [start] if start == goal else []
    out = [start]
    for e in walk:
        a, b = e["source"], e["target"]
        out.append(b if a == out[-1] else a)
    return out
