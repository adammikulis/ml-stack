"""A graph that outlives the process.

Every test here opens a real store on disk and, where it matters, closes it and opens it
again on a fresh handle: a count read back from the object that just wrote it proves nothing
about what was persisted.
"""

import pytest

from ml_stack.graph.store import GraphStore

pytest.importorskip("ladybug", reason="the store needs ml-stack[store]")

GRAPH = {
    "nodes": [
        {"id": "person:ada", "kind": "person", "label": "Ada Lovelace", "mentions": 4,
         "attrs": {"role": "analyst", "member": True}},
        {"id": "person:bea", "kind": "person", "label": "Bea Marlow", "mentions": 2, "attrs": {}},
        {"id": "topic:compilers", "kind": "topic", "label": "compilers", "mentions": 3, "attrs": {}},
        {"id": "place:turin", "kind": "place", "label": "Turin", "mentions": 1, "attrs": {}},
        {"id": "org:pellard", "kind": "org", "label": "Pellard Foundry", "mentions": 1, "attrs": {}},
    ],
    "edges": [
        {"source": "person:ada", "target": "topic:compilers", "rel": "interested_in", "weight": 3,
         "messages": ["C1-1.1"]},
        {"source": "person:bea", "target": "topic:compilers", "rel": "interested_in", "weight": 2},
        {"source": "person:ada", "target": "place:turin", "rel": "based_in", "weight": 1},
    ],
}


def test_what_was_written_is_there_after_reopening(tmp_path):
    with GraphStore(tmp_path / "g") as store:
        assert store.write(GRAPH) == {"nodes": 5, "edges": 3}

    with GraphStore(tmp_path / "g") as reopened:
        back = reopened.read()
    assert [n["id"] for n in back["nodes"]] == sorted(n["id"] for n in GRAPH["nodes"])
    ada = next(n for n in back["nodes"] if n["id"] == "person:ada")
    assert ada["label"] == "Ada Lovelace" and ada["mentions"] == 4
    assert ada["attrs"] == {"role": "analyst", "member": True}
    edge = next(e for e in back["edges"] if e["target"] == "topic:compilers" and e["source"] == "person:ada")
    assert edge["rel"] == "interested_in" and edge["weight"] == 3
    assert edge["messages"] == ["C1-1.1"]


def test_an_edge_with_a_missing_end_is_refused_not_invented(tmp_path):
    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        assert store.upsert_edge({"source": "person:ada", "target": "person:nobody",
                                  "rel": "knows", "weight": 1}) is False
        assert len(store.edges()) == 3


def test_writing_twice_updates_rather_than_doubles(tmp_path):
    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        store.write({**GRAPH, "nodes": [{**GRAPH["nodes"][0], "mentions": 9}]})
        assert len(store.nodes()) == 5
        assert len(store.edges()) == 3
    with GraphStore(tmp_path / "g") as reopened:
        assert next(n for n in reopened.nodes() if n["id"] == "person:ada")["mentions"] == 9


def test_neighbours_reach_both_ways_along_an_edge(tmp_path):
    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        assert {n["id"] for n in store.neighbours("person:ada")} == {"topic:compilers", "place:turin"}
        # the edge points at the topic, and the topic still knows who is joined to it
        assert {n["id"] for n in store.neighbours("topic:compilers")} == {"person:ada", "person:bea"}


def test_the_engine_finds_the_way_between_two_people(tmp_path):
    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        assert store.shortest_path("person:ada", "person:bea") == ["person:ada", "topic:compilers", "person:bea"]
        assert store.shortest_path("person:ada", "person:ada") == ["person:ada"]
        assert store.shortest_path("person:ada", "person:nobody") == []


def test_dropping_a_node_takes_its_edges_with_it(tmp_path):
    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        assert store.drop(["topic:compilers", "person:nobody"]) == 1
    with GraphStore(tmp_path / "g") as reopened:
        assert [n["id"] for n in reopened.nodes()] == ["org:pellard", "person:ada", "person:bea", "place:turin"]
        assert [e["rel"] for e in reopened.edges()] == ["based_in"]


def test_any_question_can_be_asked_in_cypher(tmp_path):
    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        rows = store.query(
            "MATCH (a:Node)-[e:Edge]->(b:Node {kind:'topic'}) "
            "RETURN a.label AS who, b.label AS what ORDER BY who")
        assert rows == [{"who": "Ada Lovelace", "what": "compilers"},
                        {"who": "Bea Marlow", "what": "compilers"}]
        assert [n["id"] for n in store.nodes(kind="place")] == ["place:turin"]


def test_a_node_keeps_what_it_carries_beyond_the_columns(tmp_path):
    """A node's own messages are the evidence for it; losing them loses the point of it."""
    with GraphStore(tmp_path / "g") as store:
        store.write({"nodes": [{"id": "person:ada", "kind": "person", "label": "Ada Lovelace",
                                "mentions": 1, "attrs": {"role": "analyst"},
                                "messages": ["C1-1.1", "C1-1.2"], "session": True}],
                     "edges": []})
    with GraphStore(tmp_path / "g") as reopened:
        ada = reopened.nodes()[0]
    assert ada["messages"] == ["C1-1.1", "C1-1.2"]
    assert ada["session"] is True
    assert ada["attrs"] == {"role": "analyst"}


def test_what_is_about_the_graph_is_kept_with_it(tmp_path):
    graph = {**GRAPH, "stats": {"nodes": 5, "edges": 3},
             "meta": {"built_at": "2026-08-31T10:00:00"},
             "messages": {"C1-1.1": {"text": "Hello", "read": {"model": "a-model.gguf"}}}}
    with GraphStore(tmp_path / "g") as store:
        store.write(graph)
    with GraphStore(tmp_path / "g") as reopened:
        back = reopened.read()
        assert reopened.get_doc("stats") == {"nodes": 5, "edges": 3}
        assert reopened.get_doc("nowhere", "fallback") == "fallback"
    assert back["meta"]["built_at"] == "2026-08-31T10:00:00"
    assert back["messages"]["C1-1.1"]["read"]["model"] == "a-model.gguf"
    assert {n["id"] for n in back["nodes"]} == {n["id"] for n in GRAPH["nodes"]}


def test_a_store_can_be_snapshotted_and_rolled_back(tmp_path):
    """The real thing this exists for: a rebuild that goes wrong is not the end of the graph."""
    from ml_stack.graph.store import count_store, roll_back, snapshot

    path = tmp_path / "g"
    with GraphStore(path) as store:
        store.write(GRAPH)
    assert count_store(path) == {"nodes": 5, "edges": 3, "docs": 0}

    kept = snapshot(path, reason="before a rebuild")
    assert kept.counts == {"nodes": 5, "edges": 3, "docs": 0}

    with GraphStore(path) as store:      # the rebuild goes wrong
        store.drop([n["id"] for n in GRAPH["nodes"]])
    assert count_store(path)["nodes"] == 0

    roll_back(kept.path)
    assert count_store(path) == {"nodes": 5, "edges": 3, "docs": 0}
    with GraphStore(path) as store:
        assert {n["id"] for n in store.nodes()} == {n["id"] for n in GRAPH["nodes"]}


def test_a_node_can_be_found_by_what_it_means(tmp_path):
    """The writer only writes. Retrieval reads read-only, and cannot build the index itself,
    so an embedding written without one would be invisible for ever."""
    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        store.set_embedding("topic:compilers", [1.0, 0.0, 0.0], model="m")
        store.set_embedding("place:turin", [0.0, 1.0, 0.0], model="m")

    with GraphStore(tmp_path / "g", read_only=True) as reader:
        near = reader.similar([0.95, 0.05, 0.0], model="m")
    assert [n["id"] for n in near] == ["topic:compilers", "place:turin"]
    assert near[0]["label"] == "compilers"
    assert near[0]["similarity"] > near[1]["similarity"]


def test_an_embedding_written_twice_is_replaced_not_doubled(tmp_path):
    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        store.set_embedding("topic:compilers", [1.0, 0.0, 0.0], model="m")
        store.set_embedding("topic:compilers", [0.0, 0.0, 1.0], model="m")
        near = store.similar([0.0, 0.0, 1.0], model="m")
    assert [n["id"] for n in near] == ["topic:compilers"]


def test_a_node_can_be_found_by_a_word_that_is_not_quite_the_word(tmp_path):
    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        found = store.search("compiler")           # the label says "compilers"
    assert [r["id"] for r in found] == ["topic:compilers"]


def test_a_file_is_kept_with_the_node_it_belongs_to(tmp_path):
    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        store.add_asset("a1", "person:ada", b"\x00\x01binary", mime="application/octet-stream",
                        meta={"why": "a portrait"})
    with GraphStore(tmp_path / "g") as reopened:
        got = reopened.asset("a1")
        assert got["bytes"] == b"\x00\x01binary"
        assert got["node_id"] == "person:ada" and got["meta"] == {"why": "a portrait"}
        assert reopened.assets_of("person:ada") == ["a1"]
        assert reopened.asset("nothing") is None


def test_folding_one_node_into_another_moves_what_was_joined_to_it(tmp_path):
    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        # Bea is the same person as Ada, as far as this test is concerned
        assert store.merge_nodes("person:ada", "person:bea") == 1
        assert [n["id"] for n in store.nodes()] == ["org:pellard", "person:ada",
                                                    "place:turin", "topic:compilers"]
        joined = {(e["source"], e["rel"], e["target"]) for e in store.edges()}
    assert ("person:ada", "interested_in", "topic:compilers") in joined
    assert not any(e[0] == "person:bea" or e[2] == "person:bea" for e in joined)
    assert len(joined) == 2   # ada->compilers and ada->turin
