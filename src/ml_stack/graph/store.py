"""A graph that outlives the process, in one file, queryable in Cypher.

Every project that builds a graph ends up rewriting the same three things: somewhere to put it,
a way to ask what is joined to what, and a way to get it back. Ladybug is an embedded property
graph — one directory on disk, no server, native Cypher, and shortest paths in the engine rather
than in a loop here. This is the thin part on top: nodes and edges as plain dictionaries, so a
caller keeps whatever shape it already had and pays nothing to store it.

Attributes travel as JSON in a single column. A property graph could hold them as columns, but
then every project's own vocabulary becomes schema, and a schema is the thing nobody wants to
migrate.

    with GraphStore(path) as g:
        g.write(graph)                       # a whole graph, as built
        g.neighbours("person:ada")           # what it is joined to
        g.shortest_path("person:ada", "person:bea")   # how two of them connect
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

NODE_TABLE = """CREATE NODE TABLE IF NOT EXISTS Node(
    id STRING, kind STRING, label STRING, mentions INT64, attrs STRING, PRIMARY KEY (id))"""
EDGE_TABLE = """CREATE REL TABLE IF NOT EXISTS Edge(
    FROM Node TO Node, rel STRING, weight INT64, data STRING)"""
MAX_HOPS = 6


class GraphStoreUnavailable(RuntimeError):
    """Ladybug is not installed. `pip install ml-stack[store]`."""


def _json(value: Any) -> str:
    try:
        return json.dumps(value or {}, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return "{}"


def _unjson(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


class GraphStore:
    """Nodes and edges on disk, asked about in Cypher."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        try:
            import ladybug as lb
        except ImportError as exc:  # pragma: no cover - depends on what is installed
            raise GraphStoreUnavailable(str(exc)) from exc
        self.path = Path(path).expanduser()
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = lb.Database(str(self.path), read_only=read_only)
        self._conn = lb.Connection(self._db)
        if not read_only:
            self._conn.execute(NODE_TABLE)
            self._conn.execute(EDGE_TABLE)

    # -- lifetime

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        for handle in (getattr(self, "_conn", None), getattr(self, "_db", None)):
            try:
                handle.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001 - closing twice is not worth an error
                pass

    # -- asking

    def query(self, cypher: str, params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Any Cypher, as a list of rows keyed by what the query returned."""
        result = self._conn.execute(cypher, dict(params or {}))
        if isinstance(result, list):  # a multi-statement query returns one result each
            result = result[-1]
        names = result.get_column_names()
        return [dict(zip(names, row)) for row in result.get_all()]

    # -- writing

    def upsert_node(self, node: Mapping[str, Any]) -> None:
        """Put one node in, or update the one already there. ``id`` is what makes it the same."""
        self._conn.execute(
            "MERGE (n:Node {id: $id}) SET n.kind=$kind, n.label=$label, "
            "n.mentions=$mentions, n.attrs=$attrs",
            {"id": str(node["id"]), "kind": str(node.get("kind") or ""),
             "label": str(node.get("label") or ""), "mentions": int(node.get("mentions") or 0),
             "attrs": _json(node.get("attrs"))})

    def upsert_edge(self, edge: Mapping[str, Any]) -> bool:
        """Put one edge in. False when either end is not in the store."""
        rows = self.query(
            "MATCH (a:Node {id:$s}), (b:Node {id:$t}) "
            "MERGE (a)-[e:Edge {rel:$rel}]->(b) SET e.weight=$weight, e.data=$data "
            "RETURN e.rel AS rel",
            {"s": str(edge["source"]), "t": str(edge["target"]), "rel": str(edge.get("rel") or ""),
             "weight": int(edge.get("weight") or 1),
             "data": _json({k: v for k, v in edge.items()
                            if k not in ("source", "target", "rel", "weight")})})
        return bool(rows)

    def write(self, graph: Mapping[str, Any]) -> dict[str, int]:
        """A whole graph as built. Nodes first, so no edge arrives before its ends."""
        nodes = list(graph.get("nodes") or ())
        for node in nodes:
            self.upsert_node(node)
        kept = sum(1 for edge in (graph.get("edges") or ()) if self.upsert_edge(edge))
        return {"nodes": len(nodes), "edges": kept}

    # -- reading back

    def nodes(self, kind: str | None = None) -> list[dict[str, Any]]:
        rows = self.query(
            "MATCH (n:Node) " + ("WHERE n.kind = $kind " if kind else "")
            + "RETURN n.id AS id, n.kind AS kind, n.label AS label, "
              "n.mentions AS mentions, n.attrs AS attrs ORDER BY n.id",
            {"kind": kind} if kind else None)
        return [{**r, "attrs": _unjson(r["attrs"])} for r in rows]

    def edges(self, rel: str | None = None) -> list[dict[str, Any]]:
        rows = self.query(
            "MATCH (a:Node)-[e:Edge]->(b:Node) " + ("WHERE e.rel = $rel " if rel else "")
            + "RETURN a.id AS source, e.rel AS rel, b.id AS target, "
              "e.weight AS weight, e.data AS data ORDER BY a.id, e.rel, b.id",
            {"rel": rel} if rel else None)
        return [{**{k: v for k, v in r.items() if k != "data"}, **_unjson(r["data"])} for r in rows]

    def read(self) -> dict[str, Any]:
        """The graph in the shape it went in as."""
        return {"nodes": self.nodes(), "edges": self.edges()}

    def neighbours(self, node_id: str) -> list[dict[str, Any]]:
        """Everything joined to this node, whichever way the edge points."""
        return self.query(
            "MATCH (a:Node {id:$id})-[e:Edge]-(b:Node) "
            "RETURN b.id AS id, b.label AS label, b.kind AS kind, e.rel AS rel, "
            "e.weight AS weight ORDER BY e.weight DESC, b.id",
            {"id": node_id})

    def shortest_path(self, start: str, goal: str, *, hops: int = MAX_HOPS) -> list[str]:
        """The ids along the shortest way across, ends included. Empty when there is none.

        The engine walks it, not a loop here, and it counts hops rather than weighing them: for
        a path chosen on how well attested each link is, see ``ml_stack.entities.paths``.
        """
        if start == goal:
            return [start] if self.query("MATCH (n:Node {id:$id}) RETURN n.id", {"id": start}) else []
        rows = self.query(
            f"MATCH p = (a:Node {{id:$s}})-[e:Edge* SHORTEST 1..{int(hops)}]-(b:Node {{id:$t}}) "
            "RETURN nodes(p) AS walked",
            {"s": start, "t": goal})
        if not rows:
            return []
        # the engine hands back whole nodes; only the ids are wanted here
        return [n["id"] for n in rows[0]["walked"]]

    def drop(self, node_ids: Iterable[str]) -> int:
        """Take nodes out, and every edge that touched them."""
        gone = 0
        for node_id in node_ids:
            if self.query("MATCH (n:Node {id:$id}) RETURN n.id", {"id": node_id}):
                self._conn.execute("MATCH (n:Node {id:$id}) DETACH DELETE n", {"id": node_id})
                gone += 1
        return gone
