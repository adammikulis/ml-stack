"""Graphs as tensors: a container, message passing, DAG sweeps, topology construction.

Lab tier.

Deliberately **not** a graph-algorithm library. Shortest path, connected components,
community detection, centrality -- use ``networkx`` via ``Graph.to_networkx()``. It has had
those right for twenty years and reimplementing them here would be strictly worse.

What is here is the part networkx cannot do: keeping a graph in the array backend so it can
go through a model, and moving values along its edges inside an autograd graph.
"""

from __future__ import annotations

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
]
