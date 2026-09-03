"""``ml-stack-store``: ``check`` and ``docs``, run in-process on stores built in tmp_path.

Nothing under ``~/.ml-stack`` is touched, and no real graph is read.
"""

import json

import pytest

from ml_stack.graph.store import GraphStore
from ml_stack.graph.store_cli import main

pytest.importorskip("ladybug", reason="the store needs ml-stack[store]")

GRAPH = {
    "nodes": [
        {"id": "person:ada", "kind": "person", "label": "Ada Lovelace", "mentions": 2, "attrs": {}},
        {"id": "topic:compilers", "kind": "topic", "label": "compilers", "mentions": 1, "attrs": {}},
    ],
    "edges": [{"source": "person:ada", "target": "topic:compilers", "rel": "interested_in",
               "weight": 2}],
}


def a_store(tmp_path, docs):
    path = tmp_path / "g"
    with GraphStore(path) as store:
        store.write({**GRAPH, **docs})
    return path


def test_a_consistent_store_checks_clean_and_exits_0(tmp_path, capsys):
    path = a_store(tmp_path, {"stats": {"nodes": 2, "edges": 1}})
    assert main(["check", str(path)]) == 0
    assert capsys.readouterr().out == f"{path}: clean\n"


def test_docs_lists_every_document_with_its_size(tmp_path, capsys):
    stats, meta = {"nodes": 2, "edges": 1}, {"built_at": "2026-01-01T00:00:00"}
    path = a_store(tmp_path, {"stats": stats, "meta": meta})
    assert main(["docs", str(path)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"_schema\t{len(json.dumps({'version': 2}))} chars",
        f"meta\t{len(json.dumps(meta))} chars",
        f"stats\t{len(json.dumps(stats))} chars",
        "3 docs",
    ]


def test_a_document_a_scan_reads_empty_is_reported_then_fixed(tmp_path, capsys, monkeypatch):
    """The fault as measured on 2026-09-01: '' through a scan of Doc.value, whole by key.

    The scan lies here the way that store did, and stops lying about a document once it is
    rewritten -- what ``--fix`` hopes for, and the part no fresh file has reproduced, which
    is why the command checks again afterwards rather than announcing a repair.
    """
    doc = {"label": "tried", "rows": [{"question": "who welds?"}] * 3}
    path = a_store(tmp_path, {"bench:tried": doc, "bench:kept": {"label": "kept"}})
    size = len(json.dumps(doc))
    lost = {"bench:tried"}
    honest, real_put = GraphStore.query, GraphStore.put_doc

    def as_that_store_did(self, cypher, params=None):
        rows = honest(self, cypher, params)
        if "d.value" in cypher and "{key" not in cypher:          # a scan, not a lookup
            return [{**r, "value": "" if r.get("key") in lost else r["value"]} for r in rows]
        return rows

    def put_and_heal(self, key, value):
        real_put(self, key, value)
        lost.discard(key)

    monkeypatch.setattr(GraphStore, "query", as_that_store_did)
    monkeypatch.setattr(GraphStore, "put_doc", put_and_heal)

    assert main(["check", str(path)]) == 1
    assert capsys.readouterr().out.splitlines() == [
        f"doc bench:tried: scan read 0 chars, key read {size} chars",
        f"{path}: 1 findings",
    ]
    assert main(["docs", str(path)]) == 0
    assert f"bench:tried\t{size} chars  (scan reads 0)" in capsys.readouterr().out.splitlines()

    assert main(["check", str(path), "--fix"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"doc bench:tried: scan read 0 chars, key read {size} chars",
        f"rewrote doc bench:tried ({size} chars)",
        f"{path}: rewrote 1, clean",
    ]
    assert main(["check", str(path)]) == 0
    with GraphStore(path, read_only=True) as reader:
        assert reader.get_doc("bench:tried") == doc


def test_a_document_emptied_on_disk_cannot_be_fixed_and_the_command_says_so(tmp_path, capsys):
    path = a_store(tmp_path, {"bench:tried": {"label": "tried"}})
    with GraphStore(path) as store:
        store.query("MATCH (d:Doc {key:'bench:tried'}) SET d.value = '' RETURN d.key AS key")
    line = "doc bench:tried: empty by key and by scan; nothing left to restore it from"
    assert main(["check", str(path), "--fix"]) == 1
    assert capsys.readouterr().out.splitlines() == [
        line, f"still: {line}", f"{path}: rewrote 0, 1 findings remain"]


def test_a_path_with_no_store_exits_2(tmp_path, capsys):
    assert main(["check", str(tmp_path / "nowhere")]) == 2
    assert "no store there" in capsys.readouterr().err


def test_tidy_rejudge_asks_a_served_judge_again_about_the_held_verdicts(tmp_path, capsys,
                                                                       monkeypatch):
    from ml_stack.graph.tidy import DECISIONS
    from tests.test_graph_tidy import _node, _store
    from tests.test_graph_tidy_judge import Scripted

    path = _store(tmp_path, [_node("concept:glimmer-node", "glimmer node", mentions=4),
                             _node("concept:glimer-node", "glimer node", mentions=1)], [])
    scripted = Scripted({("glimmer node", "glimer node"): "different"})
    monkeypatch.setattr("ml_stack.client.Client", lambda url, **kw: scripted)
    assert main(["tidy", str(path), "--base-url", "http://localhost:1"]) == 0
    assert "judged 0 pair(s) the same and 1 different" in capsys.readouterr().out
    assert len(scripted.calls) == 1

    scripted.verdicts[("glimmer node", "glimer node")] = "same"
    assert main(["tidy", str(path), "--base-url", "http://localhost:1"]) == 0
    assert len(scripted.calls) == 1, "held, so not asked"
    assert main(["tidy", str(path), "--base-url", "http://localhost:1", "--rejudge"]) == 0
    out = capsys.readouterr().out
    assert len(scripted.calls) == 2 and "asked the judge again about 1 held verdict(s)" in out
    with GraphStore(path, read_only=True) as store:
        assert store.get_doc(DECISIONS)["pairs"]["concept:glimer-node|concept:glimmer-node"][
            "verdict"] == "same"
        assert [n["id"] for n in store.nodes()] == ["concept:glimmer-node"]
