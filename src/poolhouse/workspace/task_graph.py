"""Native coordination graph task records and relationships."""

from typing import Any

from poolhouse.graph.store import GraphStore


def record(graph: GraphStore, ident: str, kind: str) -> dict[str, Any]:
    found = next((node['attrs'] for node in graph.nodes(kind) if node['id'] == ident), None)
    if found is None:
        raise ValueError(f'{kind} record does not exist')
    return dict(found)


def save(graph: GraphStore, kind: str, value: dict[str, Any]) -> None:
    graph.upsert_node({'id': value['id'], 'kind': kind, 'label': value.get('title', value['id']),
                       'attrs': value})


def link(graph: GraphStore, source: str, target: str, rel: str) -> None:
    if not graph.upsert_edge({'source': source, 'target': target, 'rel': rel}):
        raise ValueError('a task graph relationship refers to a missing record')
