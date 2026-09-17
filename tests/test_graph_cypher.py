"""A schema-free store: rows out of Cypher, the indexes a reader searches, and a census that a
snapshot is verified by. Every test works on a real store on disk."""

from __future__ import annotations

import pytest

from ml_stack.graph.cypher import CypherStore, census, literal
from ml_stack.graph.snapshots import take


def a_people_store(path):
    with CypherStore(path) as db:
        db.query("CREATE NODE TABLE Person(id STRING, name STRING, PRIMARY KEY(id))")
        db.query("CREATE NODE TABLE Vec(id STRING, v FLOAT[3], PRIMARY KEY(id))")
        db.query("CREATE REL TABLE Knows(FROM Person TO Person, since STRING, note STRING)")
        for pid, name in (("p1", "ada lovelace"), ("p2", "charles babbage"), ("p3", "mary somerville")):
            db.query("CREATE (:Person {id: $id, name: $name})", {"id": pid, "name": name})
        for vid, v in (("v1", [1.0, 0.0, 0.0]), ("v2", [0.0, 1.0, 0.0])):
            db.query("CREATE (:Vec {id: $id, v: $v})", {"id": vid, "v": v})
        db.query("MATCH (a:Person {id: 'p1'}), (b:Person {id: 'p2'}) "
                 "CREATE (a)-[:Knows {since: '1833', note: $n}]->(b)", {"n": "met at a party " * 5})
        db.index_words("Person", "person_words", ["name"])
        db.index_vectors("Vec", "vec_near", "v")
    return path


def test_each_word_search_on_one_handle_answers_its_own_question(tmp_path):
    with CypherStore(a_people_store(tmp_path / "p.lbug"), read_only=True) as db:
        for _ in range(3):
            for word, expect in (("ada", "p1"), ("babbage", "p2"), ("somerville", "p3")):
                found = db.search_words("Person", "person_words", word, returns="node.id AS id")
                assert [r["id"] for r in found] == [expect]


def test_a_node_written_twice_is_found_once(tmp_path):
    path = a_people_store(tmp_path / "p.lbug")
    with CypherStore(path) as db:
        for _ in range(3):
            db.query("MERGE (n:Person {id: 'p1'}) SET n.name = $name", {"name": "ada lovelace"})
        assert [r["id"] for r in db.search_words("Person", "person_words", "lovelace",
                                                 returns="node.id AS id")] == ["p1"]


def test_the_nearest_vector_comes_first_and_every_run_answers_its_own_question(tmp_path):
    with CypherStore(a_people_store(tmp_path / "p.lbug"), read_only=True) as db:
        for vector, expect in (([0.9, 0.1, 0.0], "v1"), ([0.1, 0.9, 0.0], "v2")) * 2:
            rows = db.nearest("Vec", "vec_near", vector, limit=2, returns="node.id AS id")
            assert rows[0]["id"] == expect and rows[0]["distance"] <= rows[1]["distance"]


def test_a_search_without_an_index_finds_nothing_rather_than_failing(tmp_path):
    with CypherStore(a_people_store(tmp_path / "p.lbug"), read_only=True) as db:
        assert db.search_words("Person", "no_such_index", "ada", returns="node.id AS id") == []
        assert db.index_words("Person", "another", ["name"]) is False    # a reader builds nothing


def test_an_index_can_be_dropped_so_its_table_can_go(tmp_path):
    with CypherStore(a_people_store(tmp_path / "p.lbug")) as db:
        assert db.has_index("Vec", "vec_near") and db.has_table("Vec")
        assert db.drop_index("Vec", "vec_near") is True
        db.query("DROP TABLE Vec")
        assert not db.has_table("Vec") and db.drop_index("Vec", "vec_near") is False


def test_the_census_counts_every_table_and_no_relation_reads_two_ways(tmp_path):
    assert census(a_people_store(tmp_path / "p.lbug")) == {
        "Knows": 1, "Knows disagreeing": 0, "Person": 3, "Vec": 2}


def test_a_relation_that_reads_differently_backwards_is_counted(tmp_path):
    path = a_people_store(tmp_path / "p.lbug")

    class Skewed(CypherStore):
        def _rows_by_id(self, cypher):
            for key, digest in super()._rows_by_id(cypher):
                yield key, (b"another" if "<-" in cypher else digest)

    with Skewed(path, read_only=True) as db:
        assert db.census()["Knows disagreeing"] == 1


def test_a_snapshot_verified_by_census_carries_every_table(tmp_path):
    path = a_people_store(tmp_path / "p.lbug")
    kept = take(path, reason="before a change", count=census)
    assert kept.counts == census(path) == census(kept.path)


def test_a_statement_that_will_not_prepare_raises_with_the_engines_reason(tmp_path):
    with CypherStore(a_people_store(tmp_path / "p.lbug"), read_only=True) as db, \
            pytest.raises(RuntimeError, match=r"(?i)binder|parser|exist"):
        db.query("MATCH (n:Nobody {id: $id}) RETURN n", {"id": "x"})


def test_a_value_written_into_a_statement_is_written_as_the_engine_reads_it():
    assert literal("ada") == "'ada'"
    assert literal("it's") == "'it\\'s'"
    assert literal("a\\b") == "'a\\\\b'"
    assert literal(["a", "b"]) == "['a', 'b']"
    assert literal(0.5) == "0.5" and literal(3) == "3" and literal(True) == "true"
    with pytest.raises(TypeError, match="no Cypher literal"):
        literal({"a": 1})


def test_a_path_predicate_written_with_literals_filters_what_it_names(tmp_path):
    path = tmp_path / "p.lbug"
    with CypherStore(path) as db:
        db.query("CREATE NODE TABLE N(id STRING, PRIMARY KEY(id))")
        db.query("CREATE REL TABLE R(FROM N TO N, eid STRING)")
        for node in ("A", "X", "B", "M", "N"):
            db.query("CREATE (:N {id: $id})", {"id": node})
        for src, dst in (("A", "X"), ("X", "B"), ("A", "M"), ("M", "N"), ("N", "B")):
            db.query("MATCH (a:N {id: $a}), (b:N {id: $b}) CREATE (a)-[:R {eid: $e}]->(b)",
                     {"a": src, "b": dst, "e": src + dst})
        skipped = literal(["AX", "XB"])
        rows = db.query(f"MATCH (a:N {{id: 'A'}})-[r:R* SHORTEST 1..4 (e, n | WHERE NOT e.eid IN "
                        f"{skipped})]-(b:N {{id: 'B'}}) RETURN properties(nodes(r), 'id') AS mids")
        assert [row["mids"] for row in rows] == [["M", "N"]]
