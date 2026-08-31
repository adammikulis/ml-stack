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

    with GraphStore(path) as store:      # the rebuild goes wrong, and insists
        store.drop([n["id"] for n in GRAPH["nodes"]], force=True)
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


def a_store_of(tmp_path, n):
    path = tmp_path / "g"
    with GraphStore(path) as store:
        store.write({"nodes": [{"id": f"n{i}", "kind": "topic", "label": f"t{i}",
                                "mentions": 1, "attrs": {}} for i in range(n)], "edges": []})
    return path


def test_a_write_that_would_take_most_of_the_store_is_refused(tmp_path):
    """A pipeline that read nothing produces an empty graph, which looks exactly like this."""
    from ml_stack.graph.store import WouldLoseTooMuch, count_store, replace

    path = a_store_of(tmp_path, 10)
    with pytest.raises(WouldLoseTooMuch, match="10 of 10"):
        replace(path, {"nodes": [], "edges": []})
    assert count_store(path)["nodes"] == 10, "it went ahead anyway"

    # and when it really is meant
    replace(path, {"nodes": [], "edges": []}, force=True, keep_copy=False)
    assert count_store(path)["nodes"] == 0


def test_an_ordinary_rebuild_still_goes_through(tmp_path):
    from ml_stack.graph.store import count_store, replace

    path = a_store_of(tmp_path, 10)
    keep = [{"id": f"n{i}", "kind": "topic", "label": f"t{i}", "mentions": 2, "attrs": {}}
            for i in range(9)]
    assert replace(path, {"nodes": keep, "edges": []}) == {"nodes": 9, "edges": 0}
    assert count_store(path)["nodes"] == 9


def test_a_write_that_takes_a_tenth_leaves_a_copy_behind(tmp_path):
    from ml_stack.graph.snapshots import snapshots
    from ml_stack.graph.store import replace

    path = a_store_of(tmp_path, 10)
    keep = [{"id": f"n{i}", "kind": "topic", "label": f"t{i}", "mentions": 1, "attrs": {}}
            for i in range(8)]
    replace(path, {"nodes": keep, "edges": []})
    kept = snapshots(path)
    assert kept and kept[0].counts["nodes"] == 10
    assert "before dropping 2 of 10" in kept[0].reason


def test_dropping_most_of_a_store_by_hand_is_refused_too(tmp_path):
    from ml_stack.graph.store import WouldLoseTooMuch

    path = a_store_of(tmp_path, 10)
    with GraphStore(path) as store:
        with pytest.raises(WouldLoseTooMuch):
            store.drop([f"n{i}" for i in range(9)])
        assert len(store.nodes()) == 10
        assert store.drop([f"n{i}" for i in range(9)], force=True) == 9


def test_a_store_that_does_not_exist_yet_is_simply_written(tmp_path):
    from ml_stack.graph.store import count_store, replace

    fresh = tmp_path / "new" / "g"
    assert replace(fresh, GRAPH) == {"nodes": 5, "edges": 3}
    assert count_store(fresh)["nodes"] == 5


def test_a_write_that_fails_partway_leaves_nothing(tmp_path):
    path = tmp_path / "g"
    with GraphStore(path) as store:
        store.write(GRAPH)
    with GraphStore(path) as store:
        with pytest.raises(KeyError):
            store.write({"nodes": [{"id": "person:cyd", "kind": "person", "label": "Cyd Marek",
                                    "mentions": 1, "attrs": {}},
                                   {"kind": "person"}],
                         "edges": []})
    with GraphStore(path) as reopened:
        assert {n["id"] for n in reopened.nodes()} == {n["id"] for n in GRAPH["nodes"]}


def test_a_rebuild_that_fails_leaves_the_old_graph(tmp_path):
    from ml_stack.graph.store import count_store, replace

    path = tmp_path / "g"
    with GraphStore(path) as store:
        store.write(GRAPH)
    before = count_store(path)
    with pytest.raises(ValueError):
        replace(path, {"nodes": [*GRAPH["nodes"],
                                 {"id": "person:cyd", "kind": "person", "label": "Cyd Marek",
                                  "mentions": "many", "attrs": {}}],
                       "edges": GRAPH["edges"]})
    assert count_store(path) == before
    with GraphStore(path, read_only=True) as reader:
        assert {n["id"] for n in reader.nodes()} == {n["id"] for n in GRAPH["nodes"]}


def test_a_merge_that_fails_partway_moves_nothing(tmp_path):
    path = tmp_path / "g"
    with GraphStore(path) as store:
        store.write(GRAPH)
    with GraphStore(path) as store:
        real, seen = store.upsert_edge, []

        def wobbly(edge):
            seen.append(edge)
            if len(seen) == 2:
                raise RuntimeError("the second move went wrong")
            return real(edge)

        store.upsert_edge = wobbly
        with pytest.raises(RuntimeError, match="second move"):
            store.merge_nodes("person:bea", "person:ada")
    with GraphStore(path) as reopened:
        assert {n["id"] for n in reopened.nodes()} == {n["id"] for n in GRAPH["nodes"]}
        joined = {(e["source"], e["rel"], e["target"]) for e in reopened.edges()}
    assert joined == {(e["source"], e["rel"], e["target"]) for e in GRAPH["edges"]}


def test_an_edge_can_be_removed_without_touching_its_ends(tmp_path):
    path = tmp_path / "g"
    with GraphStore(path) as store:
        store.write(GRAPH)
        assert store.remove_edge("person:ada", "topic:compilers", "interested_in") is True
        assert store.remove_edge("person:ada", "topic:compilers", "interested_in") is False
        assert store.remove_edge("person:ada", "person:nobody", "knows") is False
    with GraphStore(path) as reopened:
        joined = {(e["source"], e["rel"], e["target"]) for e in reopened.edges()}
        assert ("person:ada", "interested_in", "topic:compilers") not in joined
        assert len(joined) == 2
        assert {n["id"] for n in reopened.nodes()} == {n["id"] for n in GRAPH["nodes"]}


def test_renaming_a_node_changes_its_label_and_nothing_else(tmp_path):
    path = tmp_path / "g"
    with GraphStore(path) as store:
        store.write(GRAPH)
        assert store.rename("person:ada", "Ada of Turin") is True
        assert store.rename("person:nobody", "Nobody") is False
    with GraphStore(path) as reopened:
        ada = next(n for n in reopened.nodes() if n["id"] == "person:ada")
    assert ada["label"] == "Ada of Turin"
    assert ada["mentions"] == 4 and ada["attrs"] == {"role": "analyst", "member": True}


def test_setting_an_attribute_keeps_the_others(tmp_path):
    path = tmp_path / "g"
    with GraphStore(path) as store:
        store.write(GRAPH)
        assert store.set_attribute("person:ada", "pronouns", "she/her") is True
        assert store.set_attribute("person:nobody", "role", "ghost") is False
    with GraphStore(path) as reopened:
        ada = next(n for n in reopened.nodes() if n["id"] == "person:ada")
    assert ada["attrs"] == {"role": "analyst", "member": True, "pronouns": "she/her"}


def an_old_store_at(path):
    import ladybug as lb

    db = lb.Database(str(path))
    conn = lb.Connection(db)
    conn.execute("CREATE NODE TABLE Node(id STRING, kind STRING, label STRING, "
                 "mentions INT64, attrs STRING, PRIMARY KEY (id))")
    conn.execute("CREATE REL TABLE Edge(FROM Node TO Node, rel STRING, weight INT64)")
    conn.execute("CREATE (:Node {id:'person:ada', kind:'person', label:'Ada Lovelace', "
                 "mentions:1, attrs:'{}'})")
    conn.execute("CREATE (:Node {id:'topic:compilers', kind:'topic', label:'compilers', "
                 "mentions:1, attrs:'{}'})")
    conn.execute("MATCH (a:Node {id:'person:ada'}), (b:Node {id:'topic:compilers'}) "
                 "CREATE (a)-[:Edge {rel:'interested_in', weight:1}]->(b)")
    conn.close()
    db.close()
    return path


def test_an_old_store_is_upgraded_on_open_for_writing(tmp_path):
    path = an_old_store_at(tmp_path / "g")
    with GraphStore(path) as store:
        assert {n["id"] for n in store.nodes()} == {"person:ada", "topic:compilers"}
        assert store.counts() == {"nodes": 2, "edges": 1, "docs": 0}
        assert store.get_doc("_schema") == {"version": 2}
    with GraphStore(path, read_only=True) as reader:
        assert [e["rel"] for e in reader.edges()] == ["interested_in"]


def test_an_old_store_opened_read_only_says_it_needs_upgrading(tmp_path):
    from ml_stack.graph.store import StoreNeedsUpgrade

    path = an_old_store_at(tmp_path / "g")
    with pytest.raises(StoreNeedsUpgrade, match="older ml-stack"):
        GraphStore(path, read_only=True)


def test_the_stores_own_records_are_not_the_graphs_documents(tmp_path):
    path = tmp_path / "g"
    with GraphStore(path) as store:
        store.write(GRAPH)
    with GraphStore(path, read_only=True) as reader:
        assert reader.counts()["docs"] == 0
        assert reader.docs() == {}
        assert not any(k.startswith("_") for k in reader.read())
    with GraphStore(path) as writer:
        assert writer.get_doc("_schema") == {"version": 2}
