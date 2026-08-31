"""Graphs as tensors, and graphs on disk: a container, message passing, DAG sweeps,
topology construction, and a Cypher-queryable store that outlives the process."""

from __future__ import annotations

from ml_stack.graph.store import GraphStore, GraphStoreUnavailable
from ml_stack.graph.dag import (
    NotADAG,
    clear_cache,
    decompose_to_dags,
    require_topological_order,
    resolvent_sweep,
    topological_order,
)
from ml_stack.graph.data import (
    BatchedGraph,
    Graph,
    batch_graphs,
)
from ml_stack.graph.message import (
    degree,
    gather,
    normalize_by_degree,
    propagate,
    scatter_mean,
    scatter_sum,
)
from ml_stack.graph.topology import (
    Edges,
    build_topology,
    knn_edges,
    morton_codes,
    mst_edges,
    pairwise_distances,
    spatial_window_edges,
)

__all__ = [
    "BatchedGraph",
    "Edges",
    "Graph",
    "NotADAG",
    "batch_graphs",
    "build_topology",
    "clear_cache",
    "decompose_to_dags",
    "degree",
    "gather",
    "knn_edges",
    "morton_codes",
    "mst_edges",
    "normalize_by_degree",
    "pairwise_distances",
    "propagate",
    "require_topological_order",
    "resolvent_sweep",
    "scatter_mean",
    "scatter_sum",
    "spatial_window_edges",
    "topological_order",
    "GraphStore",
    "GraphStoreUnavailable",
]
