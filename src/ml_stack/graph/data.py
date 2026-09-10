"""A graph as arrays, so it can go through a model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Tensor = Any


@dataclass
class Graph:
    """A directed graph with optional edge weights and node features."""

    num_nodes: int
    src: Tensor
    dst: Tensor
    w: Tensor | None = None
    x: Tensor | None = None
    edge_type: Tensor | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.num_nodes) < 0:
            raise ValueError(f"num_nodes must be non-negative, got {self.num_nodes}")
        n_src, n_dst = _length(self.src), _length(self.dst)
        if n_src != n_dst:
            raise ValueError(f"src has {n_src} edges but dst has {n_dst}")
        for name in ("w", "edge_type"):
            value = getattr(self, name)
            if value is not None and _length(value) != n_src:
                raise ValueError(
                    f"{name} has {_length(value)} entries but there are {n_src} edges"
                )

    @property
    def num_edges(self) -> int:
        return _length(self.src)

    def to_networkx(self, *, directed: bool = True):
        """A networkx graph, for algorithms. Copies to host memory."""
        import networkx as nx

        graph = nx.DiGraph() if directed else nx.Graph()
        graph.add_nodes_from(range(int(self.num_nodes)))

        src, dst = _to_list(self.src), _to_list(self.dst)
        weights = _to_list(self.w) if self.w is not None else None
        for i, (u, v) in enumerate(zip(src, dst)):
            graph.add_edge(int(u), int(v), weight=float(weights[i]) if weights else 1.0)
        return graph

    @classmethod
    def from_edges(
        cls,
        num_nodes: int,
        edges: list[tuple[int, int]],
        *,
        backend: Any = None,
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> "Graph":
        """Build from a Python edge list. For tests and small graphs."""
        from ml_stack.backend import get_backend

        backend = backend or get_backend()
        ops = backend.ops
        src = ops.array([int(u) for u, _ in edges], dtype=ops.int32)
        dst = ops.array([int(v) for _, v in edges], dtype=ops.int32)
        w = ops.array([float(x) for x in weights], dtype=ops.float32) if weights else None
        return cls(num_nodes=num_nodes, src=src, dst=dst, w=w, **kwargs)


@dataclass
class BatchedGraph:
    """Several graphs merged into one super-graph with no edges between them."""

    num_nodes: int
    num_graphs: int
    src: Tensor
    dst: Tensor
    x: Tensor | None
    graph_ids: Tensor
    node_offsets: Tensor


def batch_graphs(graphs: list[Graph], *, backend: Any = None) -> BatchedGraph:
    """Merge graphs into one super-graph, offsetting each one's edge indices."""
    from ml_stack.backend import get_backend

    if not graphs:
        raise ValueError("cannot batch an empty list of graphs")

    backend = backend or get_backend()
    ops = backend.ops

    src_all: list[int] = []
    dst_all: list[int] = []
    graph_ids: list[int] = []
    offsets: list[int] = [0]
    total = 0

    for index, graph in enumerate(graphs):
        src_all += [int(v) + total for v in _to_list(graph.src)]
        dst_all += [int(v) + total for v in _to_list(graph.dst)]
        graph_ids += [index] * int(graph.num_nodes)
        total += int(graph.num_nodes)
        offsets.append(total)

    features = [g.x for g in graphs]
    stacked = ops.concatenate(features, axis=0) if all(f is not None for f in features) else None

    return BatchedGraph(
        num_nodes=total,
        num_graphs=len(graphs),
        src=ops.array(src_all, dtype=ops.int32),
        dst=ops.array(dst_all, dtype=ops.int32),
        x=stacked,
        graph_ids=ops.array(graph_ids, dtype=ops.int32),
        node_offsets=ops.array(offsets, dtype=ops.int32),
    )


def _length(value: Tensor) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return int(shape[0]) if len(shape) else 0
    return len(value)


def _to_list(value: Tensor) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)
