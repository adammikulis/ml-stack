"""Two probes a store engine bump must pass. 2026-09-03: ladybug 0.18.2 blanked other nodes'
id, attrs and data on a single DETACH DELETE in a ~10k-node store, and 0.20.2 doubled a
node written twice; 0.19.1 passes both. Measured, never trusted again."""

import pytest

from ml_stack.graph.store import GraphStore


def _big_graph(n: int = 10_000):
    nodes = [{"id": f"concept:c{i}", "kind": "concept", "label": f"glimmer node {i}",
              "mentions": 1 + i % 5, "attrs": {"definition": f"definition {i}", "aliases": []},
              "provenance": [f"lattice:1:1.{i % 40}"]} for i in range(n)]
    edges = [{"source": f"concept:c{i}", "rel": "part_of", "target": f"concept:c{(i * 7 + 1) % n}",
              "weight": 1, "provenance": [f"lattice:1:1.{i % 40}"]} for i in range(n)]
    return {"nodes": nodes, "edges": edges}


@pytest.mark.slow
def test_a_delete_at_scale_leaves_every_other_node_whole(tmp_path):
    path = tmp_path / "big.ladybug"
    with GraphStore(path) as store:
        store.write(_big_graph())
        store.drop(["concept:c300"])
        for i in range(301, 320):
            store.drop([f"concept:c{i}"])
        blank = store.query("MATCH (n:Node) WHERE n.id = '' OR n.id IS NULL RETURN count(n) AS c")
        assert blank[0]["c"] == 0, "a delete must not blank other nodes' strings"
        assert store.query("MATCH (n:Node) RETURN count(n) AS c")[0]["c"] == 10_000 - 20
        sample = store.query("MATCH (n:Node {id:$id}) RETURN n.attrs AS attrs",
                             {"id": "concept:c9000"})
        assert sample and "definition 9000" in str(sample[0]["attrs"])
        unfound = 0
        for edge in store.edges()[:2000]:
            if not store.query("MATCH (a:Node {id:$s})-[e:Edge {rel:$rel}]->(b:Node {id:$t}) "
                               "RETURN e.rel", {"s": edge["source"], "rel": edge["rel"],
                                               "t": edge["target"]}):
                unfound += 1
        assert unfound == 0, "every edge found by its ends after a delete"


def test_a_node_written_twice_is_one_node(tmp_path):
    path = tmp_path / "twice.ladybug"
    graph = {"nodes": [{"id": "topic:robotics", "kind": "topic", "label": "robotics",
                        "mentions": 1, "attrs": {}}], "edges": []}
    with GraphStore(path) as store:
        store.write(graph)
        store.write(graph)
        assert [n["id"] for n in store.nodes()] == ["topic:robotics"]
    with GraphStore(path, read_only=True) as store:
        assert [n["id"] for n in store.nodes()] == ["topic:robotics"], "and a fresh read sees it"


def _cache(store):
    """The connection's prepared-statement cache, or None when ladybug stopped keeping one."""
    return getattr(store._conn, "_pybind_implicit_prepared_cache", None)


def test_a_long_write_does_not_keep_a_plan_per_statement(tmp_path):
    n = 2_000
    graph = {"nodes": [{"id": f"concept:c{i}", "kind": "concept", "label": f"glimmer {i}",
                        "mentions": 1, "attrs": {"definition": f"definition {i}"}}
                       for i in range(n)],
             "edges": [{"source": f"concept:c{i}", "rel": "part_of",
                        "target": f"concept:c{(i * 7 + 1) % n}", "weight": 1}
                       for i in range(n)]}
    with GraphStore(tmp_path / "long.ladybug") as store:
        store.write(graph)
        held = _cache(store)
        assert held is not None, "ladybug no longer caches by statement text; re-measure _forget"
        assert len(held) < 20, (
            f"{len(held)} prepared statements kept over {2 * n} writes: every write's own "
            "text is a cache entry that is never dropped, and the store runs out of memory")
        assert store.query("MATCH (n:Node) RETURN count(n) AS c")[0]["c"] == n
        assert store.query("MATCH ()-[e:Edge]->() RETURN count(e) AS c")[0]["c"] == n
