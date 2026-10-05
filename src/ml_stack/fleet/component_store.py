"""Model sources and installed component relationships in GraphStore."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

from ml_stack import lock
from ml_stack.graph.columns import column
from ml_stack.graph.store import GraphStore


def identity(path: Path) -> str:
    return "model-file:" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()


@contextmanager
def opened(path: Path):
    with lock.only_one(path.parent / "model-components.lock", timeout=30, announce=lambda text: None):
        with GraphStore(path.parent / "model-components.db", buffer_pool_size=32 << 20,
                        max_db_size=1 << 30) as graph:
            yield graph


def read(path: Path) -> dict:
    if not (path.parent / "model-components.db").exists():
        return {}
    with opened(path) as graph:
        rows = graph.query("MATCH (m:Node {id:$id}) RETURN m.attrs AS attrs", {"id": identity(path)})
        if not rows:
            return {}
        value = column(rows[0]["attrs"], str(path))
        value["components"] = {}
        for row in graph.query("MATCH (m:Node {id:$id})-[e:Edge]->(c:Node) "
                               "RETURN e.rel AS kind, e.data AS offer", {"id": identity(path)}):
            value["components"][row["kind"]] = column(row["offer"], str(path))
        return value


def source(path: Path, reference: str) -> None:
    with opened(path) as graph:
        graph.upsert_node({"id": identity(path), "kind": "model-file", "label": path.name,
                           "attrs": {"path": str(path.resolve()), "source": reference}})


def link(model: Path, component: Path, offer: dict) -> None:
    with opened(model) as graph:
        with graph.transaction():
            for path in (model, component):
                if not graph.has(identity(path)):
                    graph.upsert_node({"id": identity(path), "kind": "model-file", "label": path.name,
                                       "attrs": {"path": str(path.resolve())}})
            graph.query("MATCH (m:Node {id:$id})-[e:Edge {rel:$kind}]->() DELETE e",
                        {"id": identity(model), "kind": offer["kind"]})
            graph.upsert_edge({"source": identity(model), "target": identity(component),
                               "rel": offer["kind"], **offer})
