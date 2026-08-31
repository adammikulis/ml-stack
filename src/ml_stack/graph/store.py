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
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

NODE_TABLE = """CREATE NODE TABLE IF NOT EXISTS Node(
    id STRING, kind STRING, label STRING, mentions INT64, attrs STRING, data STRING,
    PRIMARY KEY (id))"""
EDGE_TABLE = """CREATE REL TABLE IF NOT EXISTS Edge(
    FROM Node TO Node, rel STRING, weight INT64, data STRING)"""
# a graph is more than its nodes: where it came from, what it counts, the messages behind it
DOC_TABLE = """CREATE NODE TABLE IF NOT EXISTS Doc(
    key STRING, value STRING, PRIMARY KEY (key))"""
ASSET_TABLE = """CREATE NODE TABLE IF NOT EXISTS Asset(
    id STRING, node_id STRING, mime STRING, bytes BLOB, meta STRING, PRIMARY KEY (id))"""
NODE_COLUMNS = ("id", "kind", "label", "mentions", "attrs")
EDGE_COLUMNS = ("source", "target", "rel", "weight")
MAX_HOPS = 6

# doc keys starting "_" belong to the store itself, not to the graph
SCHEMA_VERSION = 2
SCHEMA_KEY = "_schema"


class GraphStoreUnavailable(RuntimeError):
    """Ladybug is not installed. `pip install ml-stack[store]`."""


class StoreNeedsUpgrade(RuntimeError):
    """This store was written by an older ml-stack. Open it once for writing to upgrade."""


class WouldLoseTooMuch(RuntimeError):
    """A write would have removed most of the store, which is a fault rather than a rebuild."""


# How much of a store one write may remove before it stops being a rebuild. A pipeline that
# read nothing produces an empty graph, and an empty graph looks exactly like "delete
# everything" to anything that trusts it.
MOST = 0.5
# and how much before a verified copy is taken on the way past
COPY_OVER = 0.1


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


def replace(path: str | Path, graph: Mapping[str, Any], *, force: bool = False,
            keep_copy: bool = True) -> dict[str, int]:
    """Make the store hold this graph and nothing else, safely.

    The safe part is the point. Anything no longer in the graph goes, but a write that would
    take most of the store is refused rather than performed, and one that would take a tenth
    leaves a verified copy behind first. A rebuild is a normal thing to do; losing a graph to
    one is not.
    """
    from ml_stack.graph.snapshots import take

    live = {str(n["id"]) for n in (graph.get("nodes") or ())}
    if not Path(path).expanduser().exists():
        # nothing to lose yet
        with GraphStore(path) as store:
            return store.write(graph)
    with GraphStore(path, read_only=True) as reader:
        held = [n["id"] for n in reader.nodes()]
    gone = [i for i in held if i not in live]
    if not force and held and len(gone) > len(held) * MOST:
        raise WouldLoseTooMuch(
            f"{len(gone)} of {len(held)} nodes would go in one write. If that is really meant, "
            "pass force=True; if it is not, something upstream read nothing.")
    if keep_copy and held and len(gone) > len(held) * COPY_OVER:
        take(path, reason=f"before dropping {len(gone)} of {len(held)} nodes",
             count=count_store, fold=fold_log)
    with GraphStore(path) as store:
        with store.transaction():
            store.drop(gone, force=True)  # already judged, above, against the whole store
            return store.write(graph)


def count_store(path: str | Path) -> dict[str, int]:
    """Open a store read-only on a fresh handle and count it.

    The fresh handle is the point, not an implementation detail: a bulk write can report every
    row written while reading back on the same connection, and be short when reopened. Only a
    fresh open sees what reached the disk.
    """
    with GraphStore(path, read_only=True) as store:
        return store.counts()


def fold_log(path: str | Path) -> None:
    """Open a store writable once and close it, which checkpoints its log away."""
    GraphStore(path).close()


def snapshot(path: str | Path, *, reason: str, keep: int = 10):
    """A verified copy of a store, taken before something that cannot be undone."""
    from ml_stack.graph.snapshots import take

    return take(path, reason=reason, count=count_store, fold=fold_log, keep=keep)


def roll_back(snapshot_path: str | Path):
    """Put a snapshot back, saving what is there now first."""
    from ml_stack.graph.snapshots import restore

    return restore(snapshot_path, count=count_store, fold=fold_log)


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
        self.read_only = read_only
        self._in_tx = False
        self._extensions: set[str] = set()
        self._indexed: set[str] = set()
        if not read_only:
            self._conn.execute(NODE_TABLE)
            self._conn.execute(EDGE_TABLE)
            self._conn.execute(DOC_TABLE)
            self._conn.execute(ASSET_TABLE)
            self._upgrade()
        else:
            try:
                self._require_current()
            except StoreNeedsUpgrade:
                self.close()
                raise

    def _upgrade(self) -> None:
        """Bring a store written by an older ml-stack up to the current schema, in place."""
        for table in ("Node", "Edge"):
            cols = {r["name"] for r in self.query(f"CALL TABLE_INFO('{table}') RETURN *")}
            if "data" not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD data STRING")
        if self.get_doc(SCHEMA_KEY, {}).get("version") != SCHEMA_VERSION:
            self.put_doc(SCHEMA_KEY, {"version": SCHEMA_VERSION})

    def _require_current(self) -> None:
        """Raise StoreNeedsUpgrade when a read-only open finds an older schema."""
        try:
            cols = {r["name"] for r in self.query("CALL TABLE_INFO('Node') RETURN *")}
        except RuntimeError:
            return                    # no Node table: an empty store, not an old one
        tables = {r["name"] for r in self.query("CALL SHOW_TABLES() RETURN *")}
        if "data" not in cols or "Doc" not in tables:
            raise StoreNeedsUpgrade(
                f"{self.path} was written by an older ml-stack. "
                "Open it once for writing (GraphStore(path)) to upgrade it in place.")

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

    @contextmanager
    def transaction(self):
        """Everything inside lands together, or none of it does."""
        if self._in_tx:
            yield
            return
        self._conn.execute("BEGIN TRANSACTION")
        self._in_tx = True
        try:
            yield
        except BaseException:
            try:
                self._conn.execute("ROLLBACK")
            except RuntimeError:
                pass              # a failed statement already rolled the transaction back
            raise
        else:
            self._conn.execute("COMMIT")
        finally:
            self._in_tx = False

    def upsert_node(self, node: Mapping[str, Any]) -> None:
        """Put one node in, or update the one already there. ``id`` is what makes it the same.

        Whatever the caller carries beyond the columns — the messages a node was read from, a
        flag of its own — rides along in ``data`` and comes back untouched.
        """
        self._conn.execute(
            "MERGE (n:Node {id: $id}) SET n.kind=$kind, n.label=$label, "
            "n.mentions=$mentions, n.attrs=$attrs, n.data=$data",
            {"id": str(node["id"]), "kind": str(node.get("kind") or ""),
             "label": str(node.get("label") or ""), "mentions": int(node.get("mentions") or 0),
             "attrs": _json(node.get("attrs")),
             "data": _json({k: v for k, v in node.items() if k not in NODE_COLUMNS})})

    def upsert_edge(self, edge: Mapping[str, Any]) -> bool:
        """Put one edge in. False when either end is not in the store."""
        rows = self.query(
            "MATCH (a:Node {id:$s}), (b:Node {id:$t}) "
            "MERGE (a)-[e:Edge {rel:$rel}]->(b) SET e.weight=$weight, e.data=$data "
            "RETURN e.rel AS rel",
            {"s": str(edge["source"]), "t": str(edge["target"]), "rel": str(edge.get("rel") or ""),
             "weight": int(edge.get("weight") or 1),
             "data": _json({k: v for k, v in edge.items() if k not in EDGE_COLUMNS})})
        return bool(rows)

    def write(self, graph: Mapping[str, Any]) -> dict[str, int]:
        """A whole graph as built. Nodes first, so no edge arrives before its ends.

        Anything alongside ``nodes`` and ``edges`` — what the graph counts, when it was built,
        the messages behind it — is kept as a document under its own key.
        """
        nodes = list(graph.get("nodes") or ())
        with self.transaction():
            for node in nodes:
                self.upsert_node(node)
            kept = sum(1 for edge in (graph.get("edges") or ()) if self.upsert_edge(edge))
            for key, value in graph.items():
                if key not in ("nodes", "edges"):
                    self.put_doc(key, value)
        return {"nodes": len(nodes), "edges": kept}

    def rename(self, node_id: str, label: str) -> bool:
        """Give a node a different label. False when it is not in the store."""
        if not self.query("MATCH (n:Node {id:$id}) RETURN n.id", {"id": str(node_id)}):
            return False
        self._conn.execute("MATCH (n:Node {id:$id}) SET n.label = $label",
                           {"id": str(node_id), "label": str(label)})
        return True

    def set_attribute(self, node_id: str, name: str, value: Any) -> bool:
        """Set one attribute of a node, keeping the others. False when it is not in the store."""
        rows = self.query("MATCH (n:Node {id:$id}) RETURN n.attrs AS attrs", {"id": str(node_id)})
        if not rows:
            return False
        attrs = _unjson(rows[0]["attrs"])
        attrs[str(name)] = value
        self._conn.execute("MATCH (n:Node {id:$id}) SET n.attrs = $attrs",
                           {"id": str(node_id), "attrs": _json(attrs)})
        return True

    def remove_edge(self, source: str, target: str, rel: str) -> bool:
        """Take one edge out, leaving both ends. False when it is not in the store."""
        found = self.query(
            "MATCH (a:Node {id:$s})-[e:Edge {rel:$rel}]->(b:Node {id:$t}) RETURN e.rel AS rel",
            {"s": str(source), "t": str(target), "rel": str(rel)})
        if not found:
            return False
        self._conn.execute(
            "MATCH (a:Node {id:$s})-[e:Edge {rel:$rel}]->(b:Node {id:$t}) DELETE e",
            {"s": str(source), "t": str(target), "rel": str(rel)})
        return True

    # -- reading back

    def nodes(self, kind: str | None = None) -> list[dict[str, Any]]:
        rows = self.query(
            "MATCH (n:Node) " + ("WHERE n.kind = $kind " if kind else "")
            + "RETURN n.id AS id, n.kind AS kind, n.label AS label, "
              "n.mentions AS mentions, n.attrs AS attrs, n.data AS data ORDER BY n.id",
            {"kind": kind} if kind else None)
        return [{**{k: v for k, v in r.items() if k != "data"},
                 "attrs": _unjson(r["attrs"]), **_unjson(r["data"])} for r in rows]

    def edges(self, rel: str | None = None) -> list[dict[str, Any]]:
        rows = self.query(
            "MATCH (a:Node)-[e:Edge]->(b:Node) " + ("WHERE e.rel = $rel " if rel else "")
            + "RETURN a.id AS source, e.rel AS rel, b.id AS target, "
              "e.weight AS weight, e.data AS data ORDER BY a.id, e.rel, b.id",
            {"rel": rel} if rel else None)
        return [{**{k: v for k, v in r.items() if k != "data"}, **_unjson(r["data"])} for r in rows]

    def put_doc(self, key: str, value: Any) -> None:
        """Something about the graph as a whole: where it came from, what it counts."""
        self._conn.execute("MERGE (d:Doc {key: $key}) SET d.value = $value",
                           {"key": str(key), "value": _json(value)})

    def get_doc(self, key: str, default: Any = None) -> Any:
        rows = self.query("MATCH (d:Doc {key:$key}) RETURN d.value AS value", {"key": str(key)})
        return _unjson(rows[0]["value"]) if rows else default

    def docs(self) -> dict[str, Any]:
        return {r["key"]: _unjson(r["value"])
                for r in self.query("MATCH (d:Doc) WHERE NOT d.key STARTS WITH '_' "
                                    "RETURN d.key AS key, d.value AS value")}

    def read(self) -> dict[str, Any]:
        """The graph in the shape it went in as, whole."""
        return {**self.docs(), "nodes": self.nodes(), "edges": self.edges()}

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

    def counts(self) -> dict[str, int]:
        """What is in here. What a snapshot is verified against."""
        out = {}
        for name, cypher in (("nodes", "MATCH (n:Node) RETURN count(n) AS c"),
                             ("edges", "MATCH (:Node)-[e:Edge]->(:Node) RETURN count(e) AS c"),
                             ("docs", "MATCH (d:Doc) WHERE NOT d.key STARTS WITH '_' "
                                      "RETURN count(d) AS c")):
            rows = self.query(cypher)
            if not rows:
                # a count that will not run means the store is unreadable or is not shaped the
                # way this expects. That is a store nothing can verify, never a store holding none
                raise GraphStoreUnavailable(f"could not count {name} in {self.path}")
            out[name] = int(rows[0]["c"])
        return out

    # -- searching by meaning, and by word

    def _extension(self, name: str) -> None:
        """Load an extension. Allowed on a read-only handle; only building an index is a write."""
        if name in self._extensions:
            return
        self._conn.execute(f"INSTALL {name}")
        self._conn.execute(f"LOAD EXTENSION {name}")
        self._extensions.add(name)

    def _index(self, name: str, cypher: str) -> None:
        """Build an index once per handle. A read-only handle uses whatever a writer built.

        A read-only handle attempting to create one would raise and break every search made
        through it, so it does not try.
        """
        if self.read_only or name in self._indexed:
            return
        self._indexed.add(name)
        try:
            self._conn.execute(cypher)
        except RuntimeError:
            pass                      # already there, or this build will not take the arguments

    def set_embedding(self, node_id: str, vector: Sequence[float], *, model: str = "") -> None:
        """Remember what a node means, as a vector.

        Whoever writes an embedding owns making it findable: retrieval usually reads through a
        read-only handle, which cannot build an index, so an embedding written without one is
        invisible for ever.
        """
        values = [float(x) for x in vector]
        self._conn.execute(
            f"CREATE NODE TABLE IF NOT EXISTS Embedding(id STRING, node_id STRING, "
            f"model STRING, vector FLOAT[{len(values)}], PRIMARY KEY (id))")
        self._extension("vector")
        key = f"{node_id}\u0000{model}"
        if self.query("MATCH (e:Embedding {id:$id}) RETURN e.id AS id", {"id": key}):
            self._conn.execute("MATCH (e:Embedding {id:$id}) SET e.vector = $v",
                               {"id": key, "v": values})
        else:
            self._conn.execute(
                "CREATE (e:Embedding {id:$id, node_id:$n, model:$m, vector:$v})",
                {"id": key, "n": str(node_id), "m": str(model), "v": values})
        # cosine, said explicitly, because the distance-to-similarity mapping below assumes it
        self._index("vector", "CALL CREATE_VECTOR_INDEX('Embedding', 'embedding_index', "
                              "'vector', metric := 'cosine')")

    def similar(self, vector: Sequence[float], *, model: str = "", limit: int = 10
                ) -> list[dict[str, Any]]:
        """The nodes closest in meaning to a vector, nearest first."""
        self._extension("vector")
        self._index("vector", "CALL CREATE_VECTOR_INDEX('Embedding', 'embedding_index', "
                              "'vector', metric := 'cosine')")
        try:
            rows = self.query(
                "CALL QUERY_VECTOR_INDEX('Embedding', 'embedding_index', $v, $k) "
                "RETURN node.node_id AS id, node.model AS model, distance AS distance",
                {"v": [float(x) for x in vector], "k": int(limit)})
        except RuntimeError:
            return []                 # nothing embedded yet, so nothing is close to anything
        label = {n["id"]: n["label"] for n in self.nodes()}
        out = []
        for row in rows:
            if model and row.get("model") != model:
                continue
            far = row.get("distance")
            # cosine distance runs 0 (identical) to 2 (opposite); this reads as a similarity
            near = max(0.0, min(1.0, (2.0 - float(far)) / 2.0)) if isinstance(far, (int, float)) else 0.0
            out.append({"id": row["id"], "label": label.get(row["id"], ""), "similarity": near})
        return out[:limit]

    def search(self, text: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Nodes whose label matches some words, stemmed and ranked."""
        self._extension("fts")
        self._index("fts", "CALL CREATE_FTS_INDEX('Node', 'node_index', ['label'])")
        try:
            rows = self.query(
                "CALL QUERY_FTS_INDEX('Node', 'node_index', $q, TOP := $k) "
                "RETURN node.id AS id, node.label AS label, node.kind AS kind, score AS score",
                {"q": str(text), "k": int(limit)})
        except RuntimeError:
            return []                 # no index yet, which is not the same as no match
        return rows[:limit]

    # -- files that belong to something in the graph

    def add_asset(self, asset_id: str, node_id: str, blob: bytes, *, mime: str = "",
                  meta: Mapping[str, Any] | None = None) -> None:
        """Keep a file with the node it belongs to."""
        self._conn.execute(
            "MERGE (a:Asset {id:$id}) SET a.node_id=$n, a.mime=$m, a.bytes=$b, a.meta=$meta",
            {"id": str(asset_id), "n": str(node_id), "m": str(mime), "b": bytes(blob),
             "meta": _json(meta)})

    def asset(self, asset_id: str) -> dict[str, Any] | None:
        rows = self.query("MATCH (a:Asset {id:$id}) RETURN a.node_id AS node_id, "
                          "a.mime AS mime, a.bytes AS bytes, a.meta AS meta", {"id": str(asset_id)})
        if not rows:
            return None
        row = rows[0]
        return {**row, "meta": _unjson(row["meta"]), "bytes": bytes(row["bytes"] or b"")}

    def assets_of(self, node_id: str) -> list[str]:
        return [r["id"] for r in self.query(
            "MATCH (a:Asset {node_id:$n}) RETURN a.id AS id ORDER BY a.id", {"n": str(node_id)})]

    def merge_nodes(self, keep: str, remove: str) -> int:
        """Fold one node into another, moving everything joined to it. Returns edges moved."""
        if keep == remove:
            return 0
        moved = 0
        with self.transaction():
            for edge in self.edges():
                if edge["source"] == remove and edge["target"] != keep:
                    self.upsert_edge({**edge, "source": keep})
                    moved += 1
                elif edge["target"] == remove and edge["source"] != keep:
                    self.upsert_edge({**edge, "target": keep})
                    moved += 1
            self.drop([remove])
        return moved

    def drop(self, node_ids: Iterable[str], *, force: bool = False) -> int:
        """Take nodes out, and every edge that touched them.

        Removing most of a store in one call is refused unless it is asked for outright: it is
        what a caller does when something upstream of it went wrong, and the store is the only
        copy of what it is about to lose.
        """
        wanted = [str(i) for i in node_ids]
        if not force and wanted:
            held = self.counts()["nodes"]
            if held and len(wanted) > held * MOST:
                raise WouldLoseTooMuch(
                    f"{len(wanted)} of {held} nodes would go in one write. If that is really "
                    "meant, pass force=True; if it is not, something upstream read nothing.")
        gone = 0
        with self.transaction():
            for node_id in wanted:
                if self.query("MATCH (n:Node {id:$id}) RETURN n.id", {"id": node_id}):
                    self._conn.execute("MATCH (n:Node {id:$id}) DETACH DELETE n", {"id": node_id})
                    gone += 1
        return gone
