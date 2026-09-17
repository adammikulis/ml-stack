"""What `ml_stack.graph` relies on the storage engine to do, measured against the installed one.

Each test is a property of ladybug that a module here is built around; when an upgrade turns
one red, the module that relies on it is what changes. Anything that can take the interpreter
down runs in a child process, so a crash is a red test rather than a dead worker.
"""

from __future__ import annotations

import random
import subprocess
import sys
import textwrap

import ladybug as lb
import pytest


def _child(body: str, *args: str) -> subprocess.CompletedProcess:
    script = "import sys, os\nimport ladybug as lb\n" + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", script, *args], capture_output=True, text=True,
                          timeout=300)


_WORD_INDEX = """
    conn = lb.Connection(lb.Database(sys.argv[1]))
    conn.execute("INSTALL fts")
    conn.execute("LOAD EXTENSION fts")
    conn.execute("CREATE NODE TABLE N(id STRING, label STRING, PRIMARY KEY(id))")
    conn.execute("CREATE (:N {id: 'a', label: 'hypertension'})")
    conn.execute("CREATE (:N {id: 'b', label: 'diabetes'})")
    conn.execute("CALL CREATE_FTS_INDEX('N', 'n_words', ['label'])")
    ASK = "CALL QUERY_FTS_INDEX('N', 'n_words', $q) RETURN node.id"
"""


def test_a_word_search_run_again_through_execute_crashes_the_engine(tmp_path):
    """`CypherStore._run` prepares every statement with values afresh because of this.

    Red means the engine re-runs its cached plan safely, and `_run` can go back to execute().
    """
    child = _child(_WORD_INDEX + """
    for text in ("hypertension", "diabetes"):
        print(conn.execute(ASK, {"q": text}).get_next()[0], flush=True)
    """, str(tmp_path / "again.lbug"))
    assert child.returncode != 0 and child.stdout.split() == ["a"]


def test_a_word_search_prepared_without_its_values_answers_every_run(tmp_path):
    child = _child(_WORD_INDEX + """
    import warnings
    warnings.simplefilter("ignore", DeprecationWarning)
    for text in ("hypertension", "diabetes", "hypertension", "diabetes"):
        print(conn.execute(conn.prepare(ASK), {"q": text}).get_next()[0], flush=True)
    """, str(tmp_path / "fresh.lbug"))
    assert child.returncode == 0, child.stderr[-500:]
    assert child.stdout.split() == ["a", "b", "a", "b"]


def _relation(i: int) -> tuple[str, str]:
    """A predicate, and evidence long enough to leave inline string storage."""
    return f"pred_{i % 37}", f"evidence sentence number {i} describing a long relation " * 2


def _read(conn, pattern: str) -> dict[str, tuple]:
    result = conn.execute(pattern + " RETURN CAST(id(r) AS STRING), a.id, b.id, r.edge_id, "
                                    "r.predicate, r.evidence, r.weight")
    rows = {}
    while result.has_next():
        rel, *rest = result.get_next()
        rows[rel] = tuple(rest)
    return rows


def _both_ways(path: str) -> tuple[dict, dict]:
    db = lb.Database(path, read_only=True)
    conn = lb.Connection(db)
    try:
        return (_read(conn, "MATCH (a:Entity)-[r:RELATES]->(b:Entity)"),
                _read(conn, "MATCH (b:Entity)<-[r:RELATES]-(a:Entity)"))
    finally:
        conn.close()
        db.close()


def _seeded(path: str, nodes: int = 2060, edges: int = 2100) -> None:
    rnd = random.Random(0)
    db = lb.Database(path)
    conn = lb.Connection(db)
    conn.execute("CREATE NODE TABLE Entity(id STRING, PRIMARY KEY(id))")
    conn.execute("CREATE REL TABLE RELATES(FROM Entity TO Entity, edge_id STRING, "
                 "predicate STRING, evidence STRING, weight DOUBLE)")
    conn.execute("UNWIND range(0, $n) AS i CREATE (:Entity {id: 'N' + CAST(i AS STRING)})",
                 {"n": nodes - 1})
    rows = []
    for i, (src, dst) in enumerate(sorted((rnd.randrange(nodes), rnd.randrange(nodes))
                                          for _ in range(edges))):
        predicate, evidence = _relation(i)
        rows.append({"s": f"N{src}", "o": f"N{dst}", "e": f"E{i}", "p": predicate, "ev": evidence})
    for row in rows[: edges // 2]:
        conn.execute("MATCH (s:Entity {id: $s}), (o:Entity {id: $o}) CREATE (s)-[:RELATES "
                     "{edge_id: $e, predicate: $p, evidence: $ev, weight: 1.0}]->(o)", row)
    conn.execute("UNWIND $rows AS row MATCH (s:Entity {id: row.s}), (o:Entity {id: row.o}) "
                 "CREATE (s)-[:RELATES {edge_id: row.e, predicate: row.p, evidence: row.ev, "
                 "weight: 1.0}]->(o)", {"rows": rows[edges // 2:]})
    conn.close()
    db.close()


def _write(path: str, *statements: tuple[str, dict]) -> None:
    db = lb.Database(path)
    conn = lb.Connection(db)
    for cypher, params in statements:
        conn.execute(cypher, params)
    conn.close()
    db.close()


def _agree(path: str) -> tuple[int, int]:
    forward, backward = _both_ways(path)
    assert forward.keys() == backward.keys()
    return len(forward), sum(1 for rel, row in forward.items() if backward[rel] != row)


@pytest.mark.parametrize("change", [
    ("MATCH (n:Entity {id: 'N1'}) DETACH DELETE n", {}),
    ("MATCH ()-[r:RELATES]->() WHERE r.edge_id IN ['E3', 'E700', 'E1500'] DELETE r", {}),
    ("MATCH ()-[r:RELATES]->() WHERE r.edge_id = $e SET r.predicate = $p, r.evidence = $ev",
     {"e": "E42", "p": "set_one", "ev": "evidence rewritten on one relation " * 3}),
    ("UNWIND $rows AS row MATCH ()-[r:RELATES]->() WHERE r.edge_id = row.e "
     "SET r.predicate = row.p, r.evidence = row.ev, r.weight = 0.5",
     {"rows": [{"e": f"E{i}", "p": f"bulk_{i}", "ev": f"bulk evidence {i} " * 4}
               for i in range(0, 2100, 3)]}),
    ("MATCH (s:Entity {id: 'N5'}), (o:Entity {id: 'N6'}) MERGE (s)-[r:RELATES {edge_id: 'M1'}]->(o) "
     "SET r.predicate = 'merged', r.evidence = $ev, r.weight = 2.0", {"ev": "merged evidence " * 5}),
], ids=["detach-delete", "delete-rels", "set-one", "unwind-set", "merge-set"])
def test_every_relation_reads_the_same_from_both_ends_after_a_write(tmp_path, change):
    """`CypherStore.census` counts relations that do not; a sound engine leaves none."""
    path = str(tmp_path / "both.lbug")
    _seeded(path)
    _write(path, change)
    held, disagreeing = _agree(path)
    assert held > 2000 and disagreeing == 0, f"{disagreeing} of {held} relations read differently backwards"


def test_relations_deleted_and_written_again_read_the_same_from_both_ends(tmp_path):
    path = str(tmp_path / "again.lbug")
    _seeded(path)
    gone = [f"E{i}" for i in range(1, 2100, 4)]
    again = [{"e": e, "p": "again", "ev": f"written again {e} " * 6} for e in gone]
    _write(path,
           ("UNWIND $ids AS e MATCH ()-[r:RELATES]->() WHERE r.edge_id = e DELETE r", {"ids": gone}),
           ("UNWIND $rows AS row MATCH (s:Entity {id: 'N7'}), (o:Entity {id: 'N8'}) "
            "CREATE (s)-[:RELATES {edge_id: row.e, predicate: row.p, evidence: row.ev, "
            "weight: 3.0}]->(o)", {"rows": again}))
    assert _agree(path) == (2100, 0)


def test_a_reader_opens_beside_a_writer_and_sees_what_was_committed(tmp_path):
    path = str(tmp_path / "beside.lbug")
    _write(path, ("CREATE NODE TABLE N(id INT64, PRIMARY KEY(id))", {}),
           ("UNWIND range(1, 100) AS i CREATE (:N {id: i})", {}))
    count = """
    db = lb.Database(sys.argv[1], read_only=True)
    print(lb.Connection(db).execute("MATCH (n:N) RETURN count(n)").get_next()[0])
    """
    db = lb.Database(path)
    conn = lb.Connection(db)
    try:
        conn.execute("BEGIN TRANSACTION")
        conn.execute("UNWIND range(101, 200) AS i CREATE (:N {id: i})")
        assert _child(count, path).stdout.split() == ["100"]
        conn.execute("COMMIT")
        assert _child(count, path).stdout.split() == ["200"]
        second = _child("lb.Database(sys.argv[1])\nprint('opened')", path)
        assert "opened" not in second.stdout and "lock" in second.stderr.lower()
    finally:
        conn.close()
        db.close()


def test_a_read_only_handle_never_sees_writes_made_after_it_opened(tmp_path):
    """`access.ReaderCache` reopens a cached handle when the store's files changed."""
    path = str(tmp_path / "frozen.lbug")
    _write(path, ("CREATE NODE TABLE N(id INT64, PRIMARY KEY(id))", {}),
           ("UNWIND range(1, 100) AS i CREATE (:N {id: i})", {}))
    db = lb.Database(path, read_only=True)
    conn = lb.Connection(db)
    try:
        _write(path, ("UNWIND range(101, 150) AS i CREATE (:N {id: i})", {}))
        assert conn.execute("MATCH (n:N) RETURN count(n)").get_next()[0] == 100
    finally:
        conn.close()
        db.close()


def test_a_read_only_handle_used_after_another_rewrote_the_store_crashes(tmp_path):
    """`access.read_lock` keeps writers out while a read runs because of this.

    Red means a stale handle survives a checkpoint under it, and readers need not wait.
    """
    child = _child("""
    path = sys.argv[1]
    db = lb.Database(path)
    conn = lb.Connection(db)
    conn.execute("CREATE NODE TABLE S(id INT64, t STRING, PRIMARY KEY(id))")
    conn.execute("UNWIND range(1, 200000) AS i CREATE (:S {id: i, t: 'old value ' + CAST(i AS STRING) + ' padded out to leave inline storage'})")
    conn.close(); db.close()
    reader = lb.Connection(lb.Database(path, read_only=True))
    print(reader.execute("MATCH (s:S) RETURN count(s)").get_next()[0], flush=True)
    writer_db = lb.Database(path)
    writer = lb.Connection(writer_db)
    writer.execute("MATCH (s:S) SET s.t = 'NEW ' + CAST(s.id AS STRING) + ' longer text that moves the pages around'")
    writer.execute("MATCH (s:S) WHERE s.id % 3 = 0 DELETE s")
    writer.close(); writer_db.close()
    print(len(reader.execute("MATCH (s:S) RETURN s.id, s.t").get_all()), flush=True)
    """, str(tmp_path / "stale.lbug"))
    assert child.stdout.split()[:1] == ["200000"]
    assert child.returncode != 0, "a stale reader survived a rewrite under it"


def test_a_killed_writers_log_is_read_and_a_writer_folds_it_away(tmp_path):
    """A store is two files after a crash; `snapshots.clone_store` copies and folds both."""
    path = str(tmp_path / "killed.lbug")
    _write(path, ("CREATE NODE TABLE N(id INT64, PRIMARY KEY(id))", {}),
           ("UNWIND range(1, 100) AS i CREATE (:N {id: i})", {}))
    _child("""
    conn = lb.Connection(lb.Database(sys.argv[1]))
    conn.execute("UNWIND range(101, 150) AS i CREATE (:N {id: i})")
    os._exit(0)
    """, path)
    wal = tmp_path / "killed.lbug.wal"
    assert wal.exists() and wal.stat().st_size > 0
    db = lb.Database(path, read_only=True)
    assert lb.Connection(db).execute("MATCH (n:N) RETURN count(n)").get_next()[0] == 150
    db.close()
    lb.Database(path).close()
    assert not wal.exists()
    db = lb.Database(path, read_only=True)
    assert lb.Connection(db).execute("MATCH (n:N) RETURN count(n)").get_next()[0] == 150
    db.close()


def test_a_database_that_never_loaded_fts_cannot_write_to_an_indexed_table(tmp_path):
    """`CypherStore.load` is called by every writer for every extension it may meet."""
    path = str(tmp_path / "indexed.lbug")
    db = lb.Database(path)
    conn = lb.Connection(db)
    conn.execute("CREATE NODE TABLE N(id STRING, label STRING, PRIMARY KEY(id))")
    conn.execute("CREATE (:N {id: 'n1', label: 'first'})")
    conn.execute("INSTALL fts")
    conn.execute("LOAD EXTENSION fts")
    conn.execute("CALL CREATE_FTS_INDEX('N', 'n_words', ['label'])")
    other = lb.Connection(db)
    other.execute("CREATE (:N {id: 'n2', label: 'second'})")    # the load is the database's
    conn.close()
    other.close()
    db.close()

    db = lb.Database(path)
    conn = lb.Connection(db)
    with pytest.raises(RuntimeError, match="extension is not loaded"):
        conn.execute("CREATE (:N {id: 'n3', label: 'third'})")
    conn.execute("LOAD EXTENSION fts")
    conn.execute("CREATE (:N {id: 'n3', label: 'third'})")
    conn.close()
    db.close()


def _chain(path: str, edges: int) -> None:
    _write(path, ("CREATE NODE TABLE N(id INT64, PRIMARY KEY(id))", {}),
           ("CREATE REL TABLE R(FROM N TO N, tag STRING)", {}),
           ("UNWIND range(0, $n) AS i CREATE (:N {id: i})", {"n": edges}),
           ("UNWIND range(0, $last) AS i MATCH (a:N {id: i}), (b:N {id: i + 1}) "
            "CREATE (a)-[:R {tag: 'before'}]->(b)", {"last": edges - 1}))


def _fresh_count(path: str, cypher: str) -> int:
    db = lb.Database(path, read_only=True)
    try:
        return int(lb.Connection(db).execute(cypher).get_next()[0])
    finally:
        db.close()


def test_a_bulk_relation_set_reaches_the_disk_whole_and_nowhere_else(tmp_path):
    path = str(tmp_path / "bulk.lbug")
    _chain(path, 10_000)
    _write(path, ("MATCH (a:N)-[r:R]->() WHERE a.id < 5000 SET r.tag = 'after'", {}))
    assert _fresh_count(path, "MATCH ()-[r:R]->() WHERE r.tag = 'after' RETURN count(r)") == 5000
    assert _fresh_count(path, "MATCH (a:N)-[r:R]->() WHERE a.id >= 5000 AND r.tag = 'before' "
                              "RETURN count(r)") == 5000


def test_a_recursive_patterns_predicate_takes_no_parameter(tmp_path):
    """`literal` writes values into a path predicate because of this.

    A statement is planned without its values (`CypherStore._run`), and a path predicate's
    parameter cannot be bound then: the binder cannot tell whether it belongs to the node or
    to the relationship. Red means it can, and a path predicate can carry a parameter.
    """
    db = lb.Database(str(tmp_path / "params.lbug"))
    conn = lb.Connection(db)
    conn.execute("CREATE NODE TABLE N(id STRING, PRIMARY KEY(id))")
    conn.execute("CREATE REL TABLE R(FROM N TO N, eid STRING)")
    conn.execute("CREATE (:N {id: 'A'}), (:N {id: 'B'})")
    conn.execute("MATCH (a:N {id: 'A'}), (b:N {id: 'B'}) CREATE (a)-[:R {eid: 'AB'}]->(b)")
    import warnings

    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        statement = conn.prepare("MATCH (a:N {id: 'A'})-[r:R* SHORTEST 1..4 "
                                 "(e, n | WHERE e.eid <> $skip)]-(b:N {id: 'B'}) RETURN length(r)")
        assert not statement.is_success()
        assert "does not depend on" in statement.get_error_message()
        rows = conn.execute("MATCH (a:N {id: 'A'})-[r:R* SHORTEST 1..4 (e, n | WHERE e.eid <> 'XX')]-"
                            "(b:N {id: 'B'}) RETURN length(r)").get_all()
        assert rows == [[1]]
    finally:
        conn.close()
        db.close()


def test_shortest_honours_a_node_predicate_and_a_relation_predicate(tmp_path):
    """A shortest path can avoid a node by naming it or by naming its relations.

    A - X - B is shortest; A - M - N - B avoids X, and either predicate finds it.
    """
    db = lb.Database(str(tmp_path / "shortest.lbug"))
    conn = lb.Connection(db)
    conn.execute("CREATE NODE TABLE N(id STRING, PRIMARY KEY(id))")
    conn.execute("CREATE REL TABLE R(FROM N TO N, eid STRING)")
    for node in ("A", "X", "B", "M", "N"):
        conn.execute("CREATE (:N {id: $id})", {"id": node})
    for src, dst in (("A", "X"), ("X", "B"), ("A", "M"), ("M", "N"), ("N", "B")):
        conn.execute("MATCH (a:N {id: $a}), (b:N {id: $b}) CREATE (a)-[:R {eid: $e}]->(b)",
                     {"a": src, "b": dst, "e": src + dst})

    def routes(predicate: str) -> list[list[str]]:
        result = conn.execute(f"MATCH (a:N {{id: 'A'}})-[r:R* SHORTEST 1..4 {predicate}]-"
                              "(b:N {id: 'B'}) RETURN properties(nodes(r), 'id') AS mids")
        return [row[0] for row in result.get_all()]

    try:
        assert routes("") == [["X"]]
        assert routes("(e, n | WHERE n.id <> 'X')") == [["M", "N"]]
        assert routes("(e, n | WHERE NOT e.eid IN ['AX', 'XB'])") == [["M", "N"]]
    finally:
        conn.close()
        db.close()
