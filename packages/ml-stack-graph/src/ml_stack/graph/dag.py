"""Topological order, and the resolvent sweep it enables.

On a DAG, ``(I - A)^-1 x`` -- the sum of contributions along every path -- can be computed
in one forward sweep in topological order instead of by inverting a matrix. That is the
useful thing here: an O(V+E) pass where the linear-algebra spelling is O(V^3).

The topological order itself is a host-side computation on Python ints, so it is cached.
Recomputing it means a device-to-host transfer of both edge arrays, and for a graph that
is rebuilt every forward pass but keeps its shape, that transfer dominates.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Any

from ml_stack.graph.data import Graph, _to_list

Tensor = Any

_CACHE_MAX = 64
_TOPO_CACHE: OrderedDict[tuple[int, int], list[int] | None] = OrderedDict()


class NotADAG(ValueError):
    """The graph has a cycle, so no topological order exists."""


def _edge_key(num_nodes: int, src: Tensor, dst: Tensor) -> tuple[int, int]:
    return int(num_nodes), hash((tuple(_to_list(src)), tuple(_to_list(dst))))


def topological_order(num_nodes: int, src: Tensor, dst: Tensor) -> list[int] | None:
    """Kahn's algorithm. Returns ``None`` if the graph has a cycle.

    ``None`` rather than raising, because callers routinely want to *ask* whether a graph
    is a DAG and take a different path if not; that is a question, not an error.
    """
    key = _edge_key(num_nodes, src, dst)
    if key in _TOPO_CACHE:
        _TOPO_CACHE.move_to_end(key)
        cached = _TOPO_CACHE[key]
        return list(cached) if cached is not None else None

    n = int(num_nodes)
    sources = [int(v) for v in _to_list(src)]
    targets = [int(v) for v in _to_list(dst)]

    indegree = [0] * n
    outgoing: list[list[int]] = [[] for _ in range(n)]
    for u, v in zip(sources, targets):
        indegree[v] += 1
        outgoing[u].append(v)

    queue = deque(i for i in range(n) if indegree[i] == 0)
    order: list[int] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in outgoing[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    result = order if len(order) == n else None

    _TOPO_CACHE[key] = result
    if len(_TOPO_CACHE) > _CACHE_MAX:
        _TOPO_CACHE.popitem(last=False)
    return list(result) if result is not None else None


def require_topological_order(graph: Graph) -> list[int]:
    """``topological_order`` or raise ``NotADAG``."""
    order = topological_order(graph.num_nodes, graph.src, graph.dst)
    if order is None:
        raise NotADAG(
            f"graph with {graph.num_nodes} nodes and {graph.num_edges} edges has a cycle"
        )
    return order


def resolvent_sweep(
    backend: Any,
    graph: Graph,
    x: Tensor,
    order: list[int] | None = None,
) -> Tensor:
    """``y = (I - A)^-1 x`` in one topological pass. ``x`` is ``(num_nodes, d)``.

    ``y[u] = x[u] + sum over parents p of  w(p->u) * y[p]``

    Building the result as a list of rows and stacking once at the end, rather than
    writing into ``y`` in place: an in-place write into a tensor that is also an input
    corrupts the autograd graph, and on an immutable array type it does not work at all.
    """
    ops = backend.ops
    order = order if order is not None else require_topological_order(graph)

    parents: list[list[tuple[int, float]]] = [[] for _ in range(int(graph.num_nodes))]
    weights = _to_list(graph.w) if graph.w is not None else None
    for edge, (u, v) in enumerate(zip(_to_list(graph.src), _to_list(graph.dst))):
        parents[int(v)].append((int(u), float(weights[edge]) if weights else 1.0))

    rows: list[Tensor] = [x[i] for i in range(int(graph.num_nodes))]
    for u in order:
        if not parents[u]:
            continue
        acc = rows[u]
        for p, weight in parents[u]:
            acc = acc + weight * rows[p]
        rows[u] = acc

    return ops.stack(rows, axis=0)


def decompose_to_dags(
    num_nodes: int,
    src: Tensor,
    dst: Tensor,
    *,
    directions: int = 2,
) -> list[tuple[list[int], list[int]]]:
    """Split an undirected graph into directed DAGs by BFS layering.

    Two directions gives a BFS tree oriented root-to-leaves and its reverse; four adds a
    second tree rooted at the farthest node reached by the first. This is what makes a
    sequential scan applicable to a graph -- each DAG has a valid topological order, so a
    recurrence can run over it.
    """
    n = int(num_nodes)
    sources = [int(v) for v in _to_list(src)]
    targets = [int(v) for v in _to_list(dst)]

    adjacency: list[set[int]] = [set() for _ in range(n)]
    for u, v in zip(sources, targets):
        adjacency[u].add(v)
        adjacency[v].add(u)

    def bfs_from(root: int) -> tuple[list[int], list[int]]:
        seen = [False] * n
        seen[root] = True
        queue = deque([root])
        out_src: list[int] = []
        out_dst: list[int] = []
        while queue:
            u = queue.popleft()
            for v in sorted(adjacency[u]):  # sorted: the decomposition must be stable
                if not seen[v]:
                    seen[v] = True
                    out_src.append(u)
                    out_dst.append(v)
                    queue.append(v)
        return out_src, out_dst

    if n == 0:
        return []

    forward_src, forward_dst = bfs_from(0)
    result = [(list(forward_src), list(forward_dst)), (list(forward_dst), list(forward_src))]

    if directions >= 4:
        root = forward_dst[-1] if forward_dst else min(n - 1, 1)
        second_src, second_dst = bfs_from(root)
        result += [(list(second_src), list(second_dst)), (list(second_dst), list(second_src))]

    return result


def clear_cache() -> None:
    """Drop the topological-order cache. For tests, and after a memory scare."""
    _TOPO_CACHE.clear()
