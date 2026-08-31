"""Graphs as tensors, and graphs on disk: a container, message passing, DAG sweeps,
topology construction, and a Cypher-queryable store that outlives the process."""

from __future__ import annotations

from ml_stack.graph.access import (LockError, holder, reading, release_all, write_lock,
                                   writing)
from ml_stack.graph.ask import Answer, converse, look_at, look_up, path_between
from ml_stack.graph.page import kinds_of, render, world_outline
from ml_stack.graph.search import hybrid, lexical, rrf
from ml_stack.graph.snapshots import Snapshot, SnapshotError, prune, restore, snapshots, take
from ml_stack.graph.store import (GraphStore, GraphStoreUnavailable, count_store,
                                  roll_back, snapshot)
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
    "count_store",
    "roll_back",
    "snapshot",
    "Answer",
    "converse",
    "look_at",
    "look_up",
    "path_between",
    "kinds_of",
    "render",
    "world_outline",
    "Snapshot",
    "SnapshotError",
    "prune",
    "restore",
    "snapshots",
    "take",
    "LockError",
    "holder",
    "reading",
    "release_all",
    "write_lock",
    "hybrid",
    "lexical",
    "rrf",
    "writing",
]
