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


def test_the_word_index_is_built_while_writing_so_a_reader_can_use_it(tmp_path):
    """Fails when the index is left to be built lazily by whoever searches first.

    A read-only handle cannot create an index and does not try, so retrieval — which is
    always read-only — got silence from every search, which is indistinguishable from a
    store that holds nothing. Measured on the real store: 0 through the reader, 1 through a
    writable handle, for a word that plainly matches.
    """
    path = tmp_path / "g.ladybug"
    graph = {"nodes": [{"id": "topic:robotics", "label": "robotics", "kind": "topic"},
                       {"id": "person:ada", "label": "Ada Lovelace", "kind": "person"}],
             "edges": [], "messages": {}}
    from ml_stack.graph.store import replace

    replace(path, graph)

    with GraphStore(path, read_only=True) as reader:
        rows = reader.search("robotics", limit=5)
        assert [r["id"] for r in rows] == ["topic:robotics"], "the reader found nothing"
        # it stems, which is most of why the index is worth having at all
        assert [r["id"] for r in reader.search("robot", limit=5)] == ["topic:robotics"]

    # and a store written to again is still writable, which an index on the table can break
    graph["nodes"].append({"id": "topic:swarms", "label": "swarms", "kind": "topic"})
    replace(path, graph)
    with GraphStore(path, read_only=True) as reader:
        assert [r["id"] for r in reader.search("swarms", limit=5)] == ["topic:swarms"]


def test_a_store_written_before_there_was_an_index_gets_one_on_the_next_rebuild(tmp_path):
    """Fails when only brand-new stores are indexed.

    Every store that already exists was written before this, and none of them has an index.
    A rebuild is the moment to give them one — otherwise retrieval stays silent on exactly
    the graphs people already have.
    """
    from ml_stack.graph.store import replace

    path = tmp_path / "old.ladybug"
    graph = {"nodes": [{"id": "topic:robotics", "label": "robotics", "kind": "topic"}],
             "edges": [], "messages": {}}
    # a store as one written by the old code: the index step inside a transaction is skipped
    with GraphStore(path) as store:
        with store.transaction():
            store.write(graph)
    with GraphStore(path, read_only=True) as reader:
        assert reader.search("robotics", limit=5) == [], "this store is meant to lack an index"

    graph["nodes"].append({"id": "topic:swarms", "label": "swarms", "kind": "topic"})
    replace(path, graph)
    with GraphStore(path, read_only=True) as reader:
        assert [r["id"] for r in reader.search("robotics", limit=5)] == ["topic:robotics"]


# -- a value the store cannot keep is refused, never kept as nothing ----------------------

def test_a_value_that_will_not_encode_is_refused_by_path_rather_than_kept_as_nothing(tmp_path):
    """`_json` used to return "{}" when json.dumps raised. A run with one unencodable
    field anywhere in it was then kept as an empty document, and nothing said so."""
    from ml_stack.graph.store import _json

    record = {"label": "tried", "server": {"concurrency": {"per_turn": {(0, 1): 2.0}}}}
    with pytest.raises(ValueError) as why:
        _json(record)
    assert "server.concurrency.per_turn[(0, 1)]" in str(why.value)
    assert "keys must be str" in str(why.value)

    loop: dict = {"rows": []}
    loop["rows"].append(loop)
    with pytest.raises(ValueError, match=r"rows\[0\] \(refers to itself\)"):
        _json(loop)

    with GraphStore(tmp_path / "g") as store:
        with pytest.raises(ValueError, match="per_turn"):
            store.put_doc("bench:tried", record)
        assert store.get_doc("bench:tried") is None, "nothing was kept in its place"


def test_a_plain_document_round_trips_whole(tmp_path):
    doc = {"at": "2026-01-01T00:00:00", "label": "tried", "server": {"slots": 2, "mmapped": True},
           "rows": [{"question": "who welds?", "seconds": 1.5, "shown": ["person:ada"]}],
           "unread_named": 0, "nothing": None}
    with GraphStore(tmp_path / "g") as store:
        store.put_doc("bench:tried", doc)
    with GraphStore(tmp_path / "g", read_only=True) as reader:
        assert reader.get_doc("bench:tried") == doc
        assert reader.docs() == {"bench:tried": doc}
        assert reader.doc_keys() == ["bench:tried"]


def test_docs_reads_each_value_by_key_and_never_off_a_scan(tmp_path):
    """Measured on a bench store: a scan of Doc returned '' for every value written after a
    point while a lookup by key returned all of it. The store below answers the way that
    one did -- every value blank on a scan, whole on a lookup -- and `docs()` must still
    hand back the documents."""
    doc = {"label": "tried", "rows": [{"question": "who welds?"}] * 3}
    with GraphStore(tmp_path / "g") as store:
        store.put_doc("bench:tried", doc)
        store.put_doc("bench:again", doc)

    with GraphStore(tmp_path / "g", read_only=True) as reader:
        honest = reader.query

        def as_that_store_did(cypher, params=None):
            rows = honest(cypher, params)
            if "d.value" in cypher and "{key" not in cypher:      # a scan, not a lookup
                return [{**r, **({"value": ""} if "value" in r else {})} for r in rows]
            return rows

        reader.query = as_that_store_did
        assert reader.docs() == {"bench:again": doc, "bench:tried": doc}


def test_a_document_can_be_deleted_and_says_whether_it_was_there(tmp_path):
    with GraphStore(tmp_path / "g") as store:
        store.put_doc("bench:tried", {"label": "tried"})
        store.put_doc("bench:kept", {"label": "kept"})
        assert store.delete_doc("bench:tried") is True
        assert store.delete_doc("bench:tried") is False
    with GraphStore(tmp_path / "g", read_only=True) as reader:
        assert reader.doc_keys() == ["bench:kept"]
        assert reader.counts()["docs"] == 1


# -- the store checks itself ---------------------------------------------------------------

def test_a_document_that_does_not_read_back_is_refused_at_the_write(tmp_path, monkeypatch):
    """Every put_doc reads its own document back by key, and a write the store cannot read
    back is refused rather than believed."""
    from ml_stack.graph.store import StoreMismatch

    with GraphStore(tmp_path / "g") as store:
        monkeypatch.setattr(store, "get_doc", lambda key, default=None: {"label": "other"})
        with pytest.raises(StoreMismatch, match="doc bench:tried"):
            store.put_doc("bench:tried", {"label": "tried", "rows": [1, 2]})
        monkeypatch.setattr(store, "get_doc", lambda key, default=None: default)
        with pytest.raises(StoreMismatch, match="bench:again.*nothing"):
            store.put_doc("bench:again", {"label": "again"})


def test_a_node_or_edge_that_does_not_read_back_is_refused_at_the_write(tmp_path):
    from ml_stack.graph.store import StoreMismatch

    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        honest = store.query

        def a_store_that_keeps_less(cypher, params=None):
            rows = honest(cypher, params)
            if "{id:$id}) RETURN n.id" in cypher:          # the lookup by id
                return [{**r, "label": ""} for r in rows]
            if "MERGE (a)-[e:Edge" in cypher:              # the edge's own RETURN
                return [{**r, "weight": 0} for r in rows]
            return rows

        store.query = a_store_that_keeps_less
        with pytest.raises(StoreMismatch, match="node person:ada: .*label \\(written 'Ada Lovelace', read ''\\)"):
            store.upsert_node(GRAPH["nodes"][0])
        with pytest.raises(StoreMismatch, match="edge person:ada -interested_in-> topic:compilers: .*weight \\(written 3, read 0\\)"):
            store.upsert_edge(GRAPH["edges"][0])


def test_a_rebuild_whose_count_does_not_match_is_rolled_back(tmp_path, monkeypatch):
    from ml_stack.graph.store import StoreMismatch, count_store, replace

    path = tmp_path / "g"
    with GraphStore(path) as store:
        store.write(GRAPH)
    monkeypatch.setattr(GraphStore, "counts", lambda self: {"nodes": 1, "edges": 0, "docs": 0})
    with pytest.raises(StoreMismatch, match="wrote 6 nodes and counts 1"):
        replace(path, {**GRAPH, "nodes": [*GRAPH["nodes"], {"id": "person:cyd", "kind": "person",
                                                             "label": "Cyd Marek", "mentions": 1,
                                                             "attrs": {}}]})
    monkeypatch.undo()
    assert count_store(path)["nodes"] == 5, "the refused write was kept"


def test_a_consistent_store_checks_clean(tmp_path):
    graph = {**GRAPH, "stats": {"nodes": 5, "edges": 3},
             "messages": {"C1-1.1": {"text": "Hello"}}}
    with GraphStore(tmp_path / "g") as store:
        store.write(graph)
        assert store.check() == []
    with GraphStore(tmp_path / "g", read_only=True) as reader:
        assert reader.check() == []


def test_a_document_a_scan_reads_empty_is_one_line_of_check_and_repair_rewrites_it(tmp_path):
    """The fault as measured: '' through a scan of Doc, whole by key. The scan lies here the
    way that store did, and stops lying about a document once it is rewritten -- which is
    what a rewrite hopes for, and the part no fresh file has reproduced; `check` says
    afterwards whether it took."""
    doc = {"label": "tried", "rows": [{"question": "who welds?"}] * 3}
    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        store.put_doc("bench:tried", doc)
        store.put_doc("bench:kept", doc)
        honest, lost = store.query, {"bench:tried"}

        def as_that_store_did(cypher, params=None):
            rows = honest(cypher, params)
            if "d.value" in cypher and "{key" not in cypher:      # a scan, not a lookup
                return [{**r, "value": "" if r.get("key") in lost else r["value"]} for r in rows]
            return rows

        store.query = as_that_store_did
        size = len(honest("MATCH (d:Doc {key:'bench:tried'}) RETURN d.value AS value")[0]["value"])
        assert store.check() == [f"doc bench:tried: scan read 0 chars, key read {size} chars"]

        real_put = store.put_doc

        def put_and_heal(key, value):
            real_put(key, value)
            lost.discard(key)

        store.put_doc = put_and_heal
        assert store.repair() == [f"rewrote doc bench:tried ({size} chars)"]
        assert store.check() == []
        assert store.get_doc("bench:tried") == doc


def test_a_document_emptied_on_disk_is_reported_and_cannot_be_rewritten(tmp_path):
    """`SET d.value = ''` by hand empties a document by key and by scan alike (measured:
    both read ''). The store never writes '' itself, so it is a finding; and with nothing
    whole to rewrite it from, repair leaves it for check to keep reporting."""
    with GraphStore(tmp_path / "g") as store:
        store.put_doc("bench:tried", {"label": "tried"})
        store.query("MATCH (d:Doc {key:'bench:tried'}) SET d.value = '' RETURN d.key AS key")
    with GraphStore(tmp_path / "g") as store:
        line = "doc bench:tried: empty by key and by scan; nothing left to restore it from"
        assert store.check() == [line]
        assert store.repair() == []
        assert store.check() == [line]


def test_a_node_or_edge_a_scan_reads_wrong_is_the_same_fault(tmp_path):
    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        honest = store.query

        def a_scan_that_loses_data(cypher, params=None):
            rows = honest(cypher, params)
            if "{id" not in cypher and ".data" in cypher:         # a scan of Node or Edge
                return [{**r, "data": ""} for r in rows]
            return rows

        store.query = a_scan_that_loses_data
        found = store.check()
    assert "node person:ada: scan and id disagree on data (id '{}', scan '')" in found
    assert ("edge person:ada -interested_in-> topic:compilers: scan and lookup disagree on "
            "data (lookup '{\"messages\": [\"C1-1.1\"]}', scan '')") in found
    assert len(found) == 5 + 3, found        # every node and every edge, nothing else


def test_has_says_whether_a_node_is_in_the_store(tmp_path):
    with GraphStore(tmp_path / "g") as store:
        store.write(GRAPH)
        assert store.has("person:ada") is True
        assert store.has("person:nobody") is False
        store.drop(["person:ada"])
        assert store.has("person:ada") is False
