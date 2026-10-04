"""Graph APIs loaded when their exported names are requested."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Answer": ("ml_stack.graph.answers", "Answer"),
    "Asking": ("ml_stack.asking", "Asking"),
    "BatchedGraph": ("ml_stack.graph.data", "BatchedGraph"),
    "Change": ("ml_stack.graph.propose", "Change"),
    "CypherStore": ("ml_stack.graph.cypher", "CypherStore"),
    "DOCUMENT": ("ml_stack.client.embed", "DOCUMENT"),
    "Graph": ("ml_stack.graph.data", "Graph"),
    "GraphStore": ("ml_stack.graph.store", "GraphStore"),
    "GraphStoreUnavailable": ("ml_stack.graph.cypher", "GraphStoreUnavailable"),
    "HIERARCHY": ("ml_stack.graph.relations", "HIERARCHY"),
    "LockError": ("ml_stack.graph.access", "LockError"),
    "NotADAG": ("ml_stack.graph.dag", "NotADAG"),
    "QUERY": ("ml_stack.client.embed", "QUERY"),
    "Snapshot": ("ml_stack.graph.snapshots", "Snapshot"),
    "SnapshotError": ("ml_stack.graph.snapshots", "SnapshotError"),
    "StoreNeedsUpgrade": ("ml_stack.graph.store", "StoreNeedsUpgrade"),
    "TASK": ("ml_stack.client.embed", "TASK"),
    "WouldLoseTooMuch": ("ml_stack.graph.store", "WouldLoseTooMuch"),
    "apply": ("ml_stack.graph.propose", "apply"),
    "batch_graphs": ("ml_stack.graph.data", "batch_graphs"),
    "census": ("ml_stack.graph.cypher", "census"),
    "clear_cache": ("ml_stack.graph.dag", "clear_cache"),
    "concerns": ("ml_stack.graph.concerns", "concerns"),
    "converse": ("ml_stack.graph.conversation", "converse"),
    "count_store": ("ml_stack.graph.rebuild", "count_store"),
    "cycles": ("ml_stack.graph.relations", "cycles"),
    "decompose_to_dags": ("ml_stack.graph.dag", "decompose_to_dags"),
    "degree": ("ml_stack.graph.message", "degree"),
    "embedded": ("ml_stack.graph.vectors", "embedded"),
    "gather": ("ml_stack.graph.message", "gather"),
    "geocode": ("ml_stack.graph.places", "geocode"),
    "holder": ("ml_stack.graph.access", "holder"),
    "hybrid": ("ml_stack.graph.search", "hybrid"),
    "kinds_of": ("ml_stack.graph.page", "kinds_of"),
    "lexical": ("ml_stack.graph.search", "lexical"),
    "look_around": ("ml_stack.graph.looking", "look_around"),
    "look_at": ("ml_stack.graph.looking", "look_at"),
    "look_up": ("ml_stack.graph.looking", "look_up"),
    "normalize_by_degree": ("ml_stack.graph.message", "normalize_by_degree"),
    "path_between": ("ml_stack.graph.looking", "path_between"),
    "places_in": ("ml_stack.graph.places", "places_in"),
    "points": ("ml_stack.graph.places", "points"),
    "propagate": ("ml_stack.graph.message", "propagate"),
    "proposing": ("ml_stack.graph.propose", "proposing"),
    "prune": ("ml_stack.graph.snapshots", "prune"),
    "quotes": ("ml_stack.graph.looking", "quotes"),
    "reading": ("ml_stack.graph.access", "reading"),
    "release_all": ("ml_stack.graph.access", "release_all"),
    "remember": ("ml_stack.graph.vectors", "remember"),
    "render": ("ml_stack.graph.page", "render"),
    "replace": ("ml_stack.graph.rebuild", "replace"),
    "require_topological_order": ("ml_stack.graph.dag", "require_topological_order"),
    "resolvent_sweep": ("ml_stack.graph.dag", "resolvent_sweep"),
    "resting_on": ("ml_stack.graph.drift", "resting_on"),
    "restore": ("ml_stack.graph.snapshots", "restore"),
    "roll_back": ("ml_stack.graph.rebuild", "roll_back"),
    "rrf": ("ml_stack.graph.search", "rrf"),
    "scatter_mean": ("ml_stack.graph.message", "scatter_mean"),
    "scatter_sum": ("ml_stack.graph.message", "scatter_sum"),
    "smooth": ("ml_stack.graph.vectors", "smooth"),
    "snapshot": ("ml_stack.graph.rebuild", "snapshot"),
    "snapshots": ("ml_stack.graph.snapshots", "snapshots"),
    "superseded": ("ml_stack.graph.drift", "superseded"),
    "take": ("ml_stack.graph.snapshots", "take"),
    "tensors": ("ml_stack.graph.tensors", "tensors"),
    "tools_for": ("ml_stack.graph.propose", "tools_for"),
    "topological_order": ("ml_stack.graph.dag", "topological_order"),
    "world_outline": ("ml_stack.graph.page", "world_outline"),
    "write_lock": ("ml_stack.graph.access", "write_lock"),
    "writing": ("ml_stack.graph.access", "writing"),
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
