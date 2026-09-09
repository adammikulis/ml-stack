"""What a `supersedes` edge says about a graph: which nodes trace back to a replaced
source, and what still touches them. Every fixture is invented."""

from __future__ import annotations

import pytest

from ml_stack.graph.drift import resting_on, superseded

GRAPH = {
    "nodes": [
        {"id": "source:accord-2015", "kind": "source", "label": "Trade Accord (2015) filing"},
        {"id": "source:accord-2020", "kind": "source", "label": "Trade Accord (2020) filing"},
        {"id": "concept:tariff-schedule", "kind": "concept", "label": "Tariff Schedule"},
        {"id": "concept:import-quota", "kind": "concept", "label": "Import Quota"},
        {"id": "concept:currency-clause", "kind": "concept", "label": "Currency Clause"},
        {"id": "concept:downstream-policy", "kind": "concept",
         "label": "Downstream Trade Policy"},
    ],
    "edges": [
        {"source": "source:accord-2020", "rel": "supersedes", "target": "source:accord-2015"},
        {"source": "concept:tariff-schedule", "rel": "read_from",
         "target": "source:accord-2015"},
        {"source": "concept:import-quota", "rel": "read_from", "target": "source:accord-2015"},
        {"source": "concept:currency-clause", "rel": "read_from",
         "target": "source:accord-2020"},
        {"source": "concept:downstream-policy", "rel": "requires",
         "target": "concept:tariff-schedule"},
    ],
}


# -- superseded -----------------------------------------------------------------------


def test_superseded_marks_the_replaced_source_and_everything_read_from_it():
    assert superseded(GRAPH) == {
        "source:accord-2015": "Trade Accord (2020) filing",
        "concept:tariff-schedule": "Trade Accord (2020) filing",
        "concept:import-quota": "Trade Accord (2020) filing",
    }


def test_a_node_read_from_the_surviving_source_is_not_marked():
    out = superseded(GRAPH)
    assert "concept:currency-clause" not in out
    assert "source:accord-2020" not in out


def test_a_graph_with_no_supersedes_edge_marks_nothing():
    assert superseded({"nodes": GRAPH["nodes"], "edges": []}) == {}


def test_an_empty_graph_marks_nothing():
    assert superseded({"nodes": [], "edges": []}) == {}


# -- resting_on -------------------------------------------------------------------------


def test_resting_on_finds_what_depends_on_a_superseded_node():
    out = resting_on(GRAPH, ["concept:tariff-schedule"])
    assert out == [{"id": "concept:downstream-policy", "label": "Downstream Trade Policy",
                    "rel": "requires", "on": "concept:tariff-schedule"}]


def test_resting_on_finds_every_node_whose_edge_points_at_the_wanted_one():
    out = resting_on(GRAPH, ["source:accord-2015"])
    ids = {row["id"] for row in out}
    assert ids == {"source:accord-2020", "concept:tariff-schedule", "concept:import-quota"}


def test_resting_on_is_empty_for_a_node_nothing_touches():
    assert resting_on(GRAPH, ["concept:import-quota"]) == []


def test_resting_on_an_empty_set_finds_nothing():
    assert resting_on(GRAPH, []) == []


# -- a fold and a store round trip -------------------------------------------------------


def _node(id_, label, kind="concept", mentions=1, **attrs):
    return {"id": id_, "kind": kind, "label": label, "mentions": mentions,
            "attrs": {"aliases": [], **attrs}, "provenance": [f"u:{id_}"]}


def _edge(a, rel, b, weight=1):
    return {"source": a, "rel": rel, "target": b, "weight": weight, "provenance": [f"u:{a}"]}


def test_supersedes_folds_from_its_stated_inverse_and_survives_a_store_round_trip(tmp_path):
    pytest.importorskip("ladybug", reason="the store needs ml-stack[store]")
    from ml_stack.graph.store import GraphStore
    from ml_stack.graph.tidy import canonical_direction, tidy

    assert canonical_direction("superseded_by") == ("supersedes", True)

    path = tmp_path / "g.ladybug"
    with GraphStore(path) as store:
        store.write({"nodes": [
            _node("source:accord-2015", "Trade Accord (2015) filing", kind="source"),
            _node("source:accord-2020", "Trade Accord (2020) filing", kind="source"),
            _node("concept:tariff-schedule", "Tariff Schedule"),
        ], "edges": [
            _edge("source:accord-2015", "superseded_by", "source:accord-2020"),
            _edge("concept:tariff-schedule", "read_from", "source:accord-2015"),
        ]})

    report = tidy(path, dry_run=False)
    assert report.inverses_folded == 1

    with GraphStore(path, read_only=True) as reader:
        graph = reader.read()

    found = [e for e in graph["edges"] if e["rel"] == "supersedes"]
    assert len(found) == 1
    assert {"source": "source:accord-2020", "rel": "supersedes",
           "target": "source:accord-2015", "weight": 1}.items() <= found[0].items()

    out = superseded(graph)
    assert out["source:accord-2015"] == "Trade Accord (2020) filing"
    assert out["concept:tariff-schedule"] == "Trade Accord (2020) filing"
