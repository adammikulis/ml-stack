"""Graph APIs loaded when their exported names are requested."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Answer": ("poolhouse.graph.answers", "Answer"),
    "Asking": ("poolhouse.asking", "Asking"),
    "BatchedGraph": ("poolhouse.graph.data", "BatchedGraph"),
    "Change": ("poolhouse.graph.propose", "Change"),
    "CypherStore": ("poolhouse.graph.cypher", "CypherStore"),
    "DOCUMENT": ("poolhouse.client.embed", "DOCUMENT"),
    "Graph": ("poolhouse.graph.data", "Graph"),
    "GraphStore": ("poolhouse.graph.store", "GraphStore"),
    "GraphStoreUnavailable": ("poolhouse.graph.cypher", "GraphStoreUnavailable"),
    "HIERARCHY": ("poolhouse.graph.relations", "HIERARCHY"),
    "LockError": ("poolhouse.graph.access", "LockError"),
    "NotADAG": ("poolhouse.graph.dag", "NotADAG"),
    "QUERY": ("poolhouse.client.embed", "QUERY"),
    "Snapshot": ("poolhouse.graph.snapshots", "Snapshot"),
    "SnapshotError": ("poolhouse.graph.snapshots", "SnapshotError"),
    "StoreNeedsUpgrade": ("poolhouse.graph.store", "StoreNeedsUpgrade"),
    "TASK": ("poolhouse.client.embed", "TASK"),
    "WouldLoseTooMuch": ("poolhouse.graph.store", "WouldLoseTooMuch"),
    "apply": ("poolhouse.graph.propose", "apply"),
    "batch_graphs": ("poolhouse.graph.data", "batch_graphs"),
    "census": ("poolhouse.graph.cypher", "census"),
    "clear_cache": ("poolhouse.graph.dag", "clear_cache"),
    "concerns": ("poolhouse.graph.concerns", "concerns"),
    "converse": ("poolhouse.graph.conversation", "converse"),
    "count_store": ("poolhouse.graph.rebuild", "count_store"),
    "cycles": ("poolhouse.graph.relations", "cycles"),
    "decompose_to_dags": ("poolhouse.graph.dag", "decompose_to_dags"),
    "degree": ("poolhouse.graph.message", "degree"),
    "embedded": ("poolhouse.graph.vectors", "embedded"),
    "gather": ("poolhouse.graph.message", "gather"),
    "geocode": ("poolhouse.graph.places", "geocode"),
    "holder": ("poolhouse.graph.access", "holder"),
    "hybrid": ("poolhouse.graph.search", "hybrid"),
    "kinds_of": ("poolhouse.graph.page", "kinds_of"),
    "lexical": ("poolhouse.graph.search", "lexical"),
    "look_around": ("poolhouse.graph.looking", "look_around"),
    "look_at": ("poolhouse.graph.looking", "look_at"),
    "look_up": ("poolhouse.graph.looking", "look_up"),
    "normalize_by_degree": ("poolhouse.graph.message", "normalize_by_degree"),
    "path_between": ("poolhouse.graph.looking", "path_between"),
    "places_in": ("poolhouse.graph.places", "places_in"),
    "points": ("poolhouse.graph.places", "points"),
    "propagate": ("poolhouse.graph.message", "propagate"),
    "proposing": ("poolhouse.graph.propose", "proposing"),
    "prune": ("poolhouse.graph.snapshots", "prune"),
    "quotes": ("poolhouse.graph.looking", "quotes"),
    "reading": ("poolhouse.graph.access", "reading"),
    "release_all": ("poolhouse.graph.access", "release_all"),
    "remember": ("poolhouse.graph.vectors", "remember"),
    "render": ("poolhouse.graph.page", "render"),
    "replace": ("poolhouse.graph.rebuild", "replace"),
    "require_topological_order": ("poolhouse.graph.dag", "require_topological_order"),
    "resolvent_sweep": ("poolhouse.graph.dag", "resolvent_sweep"),
    "resting_on": ("poolhouse.graph.drift", "resting_on"),
    "restore": ("poolhouse.graph.snapshots", "restore"),
    "roll_back": ("poolhouse.graph.rebuild", "roll_back"),
    "rrf": ("poolhouse.graph.search", "rrf"),
    "scatter_mean": ("poolhouse.graph.message", "scatter_mean"),
    "scatter_sum": ("poolhouse.graph.message", "scatter_sum"),
    "smooth": ("poolhouse.graph.vectors", "smooth"),
    "snapshot": ("poolhouse.graph.rebuild", "snapshot"),
    "snapshots": ("poolhouse.graph.snapshots", "snapshots"),
    "superseded": ("poolhouse.graph.drift", "superseded"),
    "take": ("poolhouse.graph.snapshots", "take"),
    "tensors": ("poolhouse.graph.tensors", "tensors"),
    "tools_for": ("poolhouse.graph.propose", "tools_for"),
    "topological_order": ("poolhouse.graph.dag", "topological_order"),
    "world_outline": ("poolhouse.graph.page", "world_outline"),
    "write_lock": ("poolhouse.graph.access", "write_lock"),
    "writing": ("poolhouse.graph.access", "writing"),
}

DOCUMENT: Any
HIERARCHY: Any
QUERY: Any
TASK: Any
Answer: Any
Asking: Any
BatchedGraph: Any
Change: Any
CypherStore: Any
Graph: Any
GraphStore: Any
GraphStoreUnavailable: Any
LockError: Any
NotADAG: Any
Snapshot: Any
SnapshotError: Any
StoreNeedsUpgrade: Any
WouldLoseTooMuch: Any
apply: Any
batch_graphs: Any
census: Any
clear_cache: Any
concerns: Any
converse: Any
count_store: Any
cycles: Any
decompose_to_dags: Any
degree: Any
embedded: Any
gather: Any
geocode: Any
holder: Any
hybrid: Any
kinds_of: Any
lexical: Any
look_around: Any
look_at: Any
look_up: Any
normalize_by_degree: Any
path_between: Any
places_in: Any
points: Any
propagate: Any
proposing: Any
prune: Any
quotes: Any
reading: Any
release_all: Any
remember: Any
render: Any
replace: Any
require_topological_order: Any
resolvent_sweep: Any
resting_on: Any
restore: Any
roll_back: Any
rrf: Any
scatter_mean: Any
scatter_sum: Any
smooth: Any
snapshot: Any
snapshots: Any
superseded: Any
take: Any
tensors: Any
tools_for: Any
topological_order: Any
world_outline: Any
write_lock: Any
writing: Any

__all__ = [
    "DOCUMENT",
    "HIERARCHY",
    "QUERY",
    "TASK",
    "Answer",
    "Asking",
    "BatchedGraph",
    "Change",
    "CypherStore",
    "Graph",
    "GraphStore",
    "GraphStoreUnavailable",
    "LockError",
    "NotADAG",
    "Snapshot",
    "SnapshotError",
    "StoreNeedsUpgrade",
    "WouldLoseTooMuch",
    "apply",
    "batch_graphs",
    "census",
    "clear_cache",
    "concerns",
    "converse",
    "count_store",
    "cycles",
    "decompose_to_dags",
    "degree",
    "embedded",
    "gather",
    "geocode",
    "holder",
    "hybrid",
    "kinds_of",
    "lexical",
    "look_around",
    "look_at",
    "look_up",
    "normalize_by_degree",
    "path_between",
    "places_in",
    "points",
    "propagate",
    "proposing",
    "prune",
    "quotes",
    "reading",
    "release_all",
    "remember",
    "render",
    "replace",
    "require_topological_order",
    "resolvent_sweep",
    "resting_on",
    "restore",
    "roll_back",
    "rrf",
    "scatter_mean",
    "scatter_sum",
    "smooth",
    "snapshot",
    "snapshots",
    "superseded",
    "take",
    "tensors",
    "tools_for",
    "topological_order",
    "world_outline",
    "write_lock",
    "writing",
]


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
