"""One-time rewrite of stored names from before the board terms were made plain.

Graph kinds ``canonical-task-ref`` and relation ``canonical-task`` became ``task-ref`` and
``task``, the evidence source ``canonical-taskboard`` became ``taskboard``, and the owner
prefix ``canonical:`` on claims and sessions became ``board:``. Each store carries a sidecar
marker file named for ``VERSION``; a store without old names is never rewritten.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from poolhouse.files import read_json, write_json
from poolhouse.graph.store import GraphStore

__all__ = ["VERSION", "marker", "migrate_claims", "migrate_graph", "opened", "rewrite_graph", "rewritten"]

VERSION = 1
KINDS = {"canonical-task-ref": "task-ref"}
RELATIONS = {"canonical-task": "task"}
SOURCES = {"canonical-taskboard": "taskboard"}
OWNER = re.compile(r"^canonical:(?=[0-9a-f]{32}:)")


def marker(path: Path) -> Path:
    """The sidecar file that says ``path`` already carries the plain names."""
    return path.with_name(f"{path.name}.names-v{VERSION}")


def rewritten(value: Any) -> Any:
    """``value`` with every old source name and owner prefix replaced, however deep."""
    if isinstance(value, str):
        return SOURCES.get(value) or OWNER.sub("board:", value)
    if isinstance(value, dict):
        return {key: rewritten(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rewritten(item) for item in value]
    return value


def rewrite_graph(graph: GraphStore) -> int:
    """Rewrite old names in place; returns how many nodes and edges changed."""
    changed = 0
    for node in graph.nodes():
        fresh = {**node, "kind": KINDS.get(node["kind"], node["kind"]), "attrs": rewritten(node["attrs"])}
        if fresh != node:
            graph.upsert_node(fresh)
            changed += 1
    for edge in graph.edges():
        if edge["rel"] in RELATIONS:
            graph.remove_edge(edge["source"], edge["rel"], edge["target"])
            graph.upsert_edge({**edge, "rel": RELATIONS[edge["rel"]]})
            changed += 1
    return changed


def migrate_graph(graph: GraphStore, path: Path) -> int:
    """Rewrite ``graph`` once; ``path`` is the store's file, which names its marker."""
    seal = marker(path)
    if seal.exists():
        return 0
    changed = rewrite_graph(graph)
    seal.write_text(str(VERSION))
    return changed


def opened(path: Path) -> GraphStore:
    """Open the file-backed graph at ``path`` with its old names already rewritten."""
    graph = GraphStore(path)
    migrate_graph(graph, path)
    return graph


def migrate_claims(path: Path) -> int:
    """Rewrite old owner prefixes in the claims file once; returns how many claims changed."""
    seal = marker(path)
    if seal.exists() or not path.exists():
        return 0
    data = read_json(path, {})
    found = data.get("claims") if isinstance(data, dict) else None
    changed = 0
    if isinstance(found, dict):
        fresh = rewritten(found)
        changed = sum(fresh[key] != found[key] for key in found)
        if changed:
            write_json(path, {**data, "claims": fresh})
            path.chmod(0o600)
    seal.write_text(str(VERSION))
    return changed
