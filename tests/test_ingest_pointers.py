"""Provenance is pointers, a fold is an upsert, and the run that read a unit is a hidden node.

Adam: "if the source already exists, it should append new nodes/connect new edges. additive";
"provenance should always be pointers to the textbook"; "a hidden node for each metadata
with hidden edges ... probably use less mem to use pointers".
"""

import json

from ml_stack import ingest
from ml_stack.graph.store import GraphStore
from tests.test_ingest import a_unit


def _read(unit, concepts, relations=(), error=""):
    return {"unit": unit.id, "source": unit.source, "chapter": unit.chapter,
            "section": unit.section, "title": unit.section_title,
            "pages": [unit.first_page, unit.last_page], "seconds": 1.0, "error": error,
            "run": "run:one",
            "extracted": {"concepts": [{"name": c, "kind": "concept", "definition": f"{c} is",
                                        "aliases": []} for c in concepts],
                          "relations": [{"from": a, "rel": r, "to": b} for a, r, b in relations],
                          "figures": [], "key_terms": []},
            "calls": []}


def _keep(tmp_path, slug, reads):
    path = tmp_path / f"sources.{slug}.reads.json"
    path.write_text(json.dumps({r["unit"]: r for r in reads}))


def test_nodes_and_edges_carry_pointers_and_nothing_copied():
    unit = a_unit()
    nodes, edges = ingest.build(_read(unit, ["glimmer node", "vault"],
                                      [("glimmer node", "part_of", "vault")])["extracted"],
                                unit, book_title="Lattice Studies")
    node = nodes["concept:glimmer-node"]
    assert node["provenance"] == [unit.id]
    assert not {"source", "chapter", "section", "page", "book_title"} & set(node["attrs"])
    assert node["attrs"]["defined_in"] == unit.id
    edge = edges[("concept:glimmer-node", "part_of", "concept:vault")]
    assert edge["provenance"] == [unit.id] and "where" not in edge


def test_a_fold_is_an_upsert_that_adds_and_never_removes(tmp_path):
    out = tmp_path / "sources"
    first = a_unit(section="1.1")
    second = a_unit(section="1.2", section_title="Vault Currents", first_page=4, last_page=5)
    _keep(tmp_path, "lattice", [_read(first, ["glimmer node", "vault"],
                                      [("glimmer node", "part_of", "vault")])])
    ingest.Progress(ingest.Progress.beside(out)).source("lattice", title="Lattice Studies",
                                                      path="l.pdf", sections=2)
    got = ingest.fold_into(out, "lattice")
    assert (got["new_nodes"], got["new_edges"]) == (2, 1)
    again = ingest.fold_into(out, "lattice")
    assert (again["new_nodes"], again["new_edges"]) == (0, 0), "idempotent"

    # a read that no longer names "vault" does not take it out of the store
    _keep(tmp_path, "lattice", [_read(first, ["glimmer node"]),
                                _read(second, ["current"], [("current", "causes", "glimmer node")])])
    more = ingest.fold_into(out, "lattice")
    assert (more["new_nodes"], more["new_edges"]) == (1, 1)
    with GraphStore(out, read_only=True) as store:
        ids = {n["id"] for n in store.nodes()}
        assert {"concept:vault", "concept:glimmer-node", "concept:current"} <= ids
        triples = {(e["source"], e["rel"], e["target"]) for e in store.edges()}
        assert ("concept:glimmer-node", "part_of", "concept:vault") in triples, "kept"
        assert ("concept:current", "causes", "concept:glimmer-node") in triples, "added"

    dry = ingest.fold_into(out, "lattice", dry_run=True)
    assert (dry["new_nodes"], dry["new_edges"]) == (0, 0)
    rebuilt = ingest.fold_into(out, "lattice", rebuild=True)
    assert rebuilt["nodes"] == 2, "the reads name two concepts now"
    with GraphStore(out, read_only=True) as store:
        assert "concept:vault" not in {n["id"] for n in store.nodes()}, \
            "rebuild is the one path that removes"


def test_the_run_node_is_hidden_and_reached_by_pointer(tmp_path):
    out = tmp_path / "sources"
    unit = a_unit()
    run_id = ingest.write_run(out, {"id": "run:one", "model": "kestrel-8B-UD-Q4_K_XL.gguf",
                                    "serving": "measured", "started": "2026-09-03T04:00:00"})
    assert run_id == "run:one"
    _keep(tmp_path, "lattice", [_read(unit, ["glimmer node", "vault"],
                                      [("glimmer node", "part_of", "vault")])])
    ingest.fold_into(out, "lattice", title="Lattice Studies")
    with GraphStore(out, read_only=True) as store:
        run = next(n for n in store.nodes(kind="run"))
        assert run["attrs"]["hidden"] is True and run["attrs"]["model"].startswith("kestrel")
        edge = next(e for e in store.edges("part_of"))
        assert ingest.origin(store, edge)[0]["model"].startswith("kestrel")
        assert ingest.origin(store, edge)[0]["units"] == [unit.id]
        where = ingest.located(store, edge)[0]
        assert where["source"] == "lattice" and where["pages"] == [2, 3]
        assert where["section"] == "1.1"


def test_hidden_nodes_stay_off_the_page_and_out_of_list_kind():
    from ml_stack.graph.ask import list_kind
    from ml_stack.graph.page import kinds_of, shown

    graph = {"nodes": [{"id": "concept:vault", "kind": "concept", "label": "vault",
                        "mentions": 2, "attrs": {}},
                       {"id": "run:one", "kind": "run", "label": "ingest",
                        "mentions": 1, "attrs": {"hidden": True}}],
             "edges": [{"source": "concept:vault", "rel": "read_by", "target": "run:one",
                        "weight": 1}]}
    assert [k["k"] for k in kinds_of(graph)] == ["concept"]
    assert [n["id"] for n in shown(graph)["nodes"]] == ["concept:vault"]
    assert shown(graph)["edges"] == []
    listed = list_kind(graph, "run")
    assert "none" in listed and "run" not in (listed.get("kinds") or {})


def test_run_record_names_what_the_run_read_with(monkeypatch):
    args = type("A", (), {"model": "kestrel-8B-UD-Q4_K_XL.gguf", "images": True, "n_max": None,
                          "temperature": None, "top_p": None, "top_k": None, "min_p": None})()
    record = ingest.run_record(args, serving="one slot, q8_0")
    assert record["model"].startswith("kestrel") and record["serving"] == "one slot, q8_0"
    assert len(record["schema_sha"]) == 16 and len(record["instructions_sha"]) == 16
    assert record["id"].startswith("run:") and record["images"] is True
    assert record["sampling"] == {"temperature": 0.1}, \
        "the resolved default, not just what a flag named"


def test_a_second_sources_names_land_on_the_first_sources_nodes_on_the_way_in(tmp_path):
    """Adam: "it will for sure re-encounter the same concepts as it learns more." The
    second source says 'vaults' and 'Glimmer Node'; both land on the first source's nodes."""
    out = tmp_path / "sources"
    first = a_unit()
    _keep(tmp_path, "lattice", [_read(first, ["glimmer node", "vault"],
                                      [("glimmer node", "part_of", "vault")])])
    ingest.Progress(ingest.Progress.beside(out)).source("lattice", title="Lattice Studies",
                                                      path="l.pdf", sections=1)
    ingest.fold_into(out, "lattice")

    second = a_unit(source="currents", book_title="Vault Currents", section="3.1",
                    section_title="Currents", first_page=9, last_page=10)
    _keep(tmp_path, "currents", [_read(second, ["vaults", "Glimmer Node", "current"],
                                       [("current", "causes", "vaults"),
                                        ("Glimmer Node", "requires", "current")])])
    ingest.Progress(ingest.Progress.beside(out)).source("currents", title="Vault Currents",
                                                      path="c.pdf", sections=1)
    got = ingest.fold_into(out, "currents")
    # 'Glimmer Node' already slugs to the first source's id, so it needs no mapping at all;
    # 'vaults' is a plural of a node the store holds and lands on it
    assert got["absorbed"]["plural"] == 1
    with GraphStore(out, read_only=True) as store:
        ids = {n["id"] for n in store.nodes()}
        assert "concept:vaults" not in ids and "concept:glimmer-node" in ids
        triples = {(e["source"], e["rel"], e["target"]) for e in store.edges()}
        assert ("concept:current", "causes", "concept:vault") in triples
        assert ("concept:glimmer-node", "requires", "concept:current") in triples
        vault = next(n for n in store.nodes() if n["id"] == "concept:vault")
        assert set(vault["provenance"]) == {first.id, second.id}


def test_fold_checks_the_store_at_its_end(tmp_path, monkeypatch, capsys):
    from ml_stack.graph.store import GraphStore

    out = tmp_path / "sources"
    unit = a_unit()
    _keep(tmp_path, "lattice", [_read(unit, ["glimmer node", "vault"],
                                      [("glimmer node", "part_of", "vault")])])
    ingest.Progress(ingest.Progress.beside(out)).source("lattice", title="Lattice Studies",
                                                      path="l.pdf", sections=1)
    assert ingest.fold(out) == 0
    assert "reads back whole" in capsys.readouterr().out
    monkeypatch.setattr(GraphStore, "check", lambda self: ["edge a -part_of-> : found by scan"])
    assert ingest.fold(out) == 1
    assert "NOT SOUND" in capsys.readouterr().out
