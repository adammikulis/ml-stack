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
        assert store.write(GRAPH) == {"nodes": 4, "edges": 3}

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
        assert len(store.nodes()) == 4
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
        assert [n["id"] for n in reopened.nodes()] == ["person:ada", "person:bea", "place:turin"]
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
    graph = {**GRAPH, "stats": {"nodes": 4, "edges": 3},
             "meta": {"built_at": "2026-08-31T10:00:00"},
             "messages": {"C1-1.1": {"text": "Hello", "read": {"model": "a-model.gguf"}}}}
    with GraphStore(tmp_path / "g") as store:
        store.write(graph)
    with GraphStore(tmp_path / "g") as reopened:
        back = reopened.read()
        assert reopened.get_doc("stats") == {"nodes": 4, "edges": 3}
        assert reopened.get_doc("nowhere", "fallback") == "fallback"
    assert back["meta"]["built_at"] == "2026-08-31T10:00:00"
    assert back["messages"]["C1-1.1"]["read"]["model"] == "a-model.gguf"
    assert {n["id"] for n in back["nodes"]} == {n["id"] for n in GRAPH["nodes"]}
