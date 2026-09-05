"""The mapping graph as arrays: node ids in index order, relations as an edge type."""

from __future__ import annotations

import pytest
from ml_stack.train.backend import available, get_backend
from ml_stack.graph.data import _to_list
from ml_stack.graph.tensors import relations_in, tensors

each_backend = pytest.mark.parametrize("name", available())

FIVE = {
    "nodes": [
        {"id": "person:ada", "kind": "person", "label": "Ada of Turin"},
        {"id": "person:bea", "kind": "person", "label": "Bea of Turin"},
        {"id": "org:quenlow", "kind": "org", "label": "Quenlow Robotics"},
        {"id": "topic:iron", "kind": "topic", "label": "iron"},
        {"id": "place:turin", "kind": "place", "label": "Turin"},
    ],
    "edges": [
        {"source": "person:ada", "target": "org:quenlow", "rel": "works_at", "weight": 3},
        {"source": "person:bea", "target": "org:quenlow", "rel": "works_at"},
        {"source": "person:ada", "target": "topic:iron", "rel": "works_on", "weight": 2},
        {"source": "org:quenlow", "target": "place:turin", "rel": "based_in", "weight": 1.5},
    ],
}


@each_backend
def test_the_ids_come_back_in_the_order_the_indices_use(name):
    made, ids = tensors(FIVE, backend=get_backend(name))
    assert ids == ["person:ada", "person:bea", "org:quenlow", "topic:iron", "place:turin"]
    assert made.num_nodes == 5
    assert made.num_edges == 4
    assert _to_list(made.src) == [0, 1, 0, 2]
    assert _to_list(made.dst) == [2, 2, 3, 4]


@each_backend
def test_an_edge_without_a_weight_carries_one(name):
    made, _ = tensors(FIVE, backend=get_backend(name))
    assert [round(w, 3) for w in _to_list(made.w)] == [3.0, 1.0, 2.0, 1.5]


@each_backend
def test_the_edge_type_indexes_the_relation_vocabulary(name):
    made, _ = tensors(FIVE, backend=get_backend(name))
    assert made.meta["relations"] == ["works_at", "works_on", "based_in"]
    assert _to_list(made.edge_type) == [0, 0, 1, 2]


@each_backend
def test_a_given_vocabulary_decides_the_numbers_and_leaves_the_rest_out(name):
    made, _ = tensors(FIVE, backend=get_backend(name), relations=["based_in", "works_on"])
    assert made.meta["relations"] == ["based_in", "works_on"]
    assert _to_list(made.edge_type) == [1, 0]
    assert made.num_edges == 2


@each_backend
def test_an_edge_naming_a_node_the_graph_does_not_hold_is_left_out(name):
    graph = {**FIVE, "edges": [*FIVE["edges"],
                               {"source": "person:ada", "target": "person:nobody",
                                "rel": "works_with"}]}
    made, _ = tensors(graph, backend=get_backend(name))
    assert made.num_edges == 4


@each_backend
def test_a_graph_with_no_edges_still_becomes_a_graph(name):
    made, ids = tensors({"nodes": FIVE["nodes"], "edges": []}, backend=get_backend(name))
    assert (made.num_nodes, made.num_edges, len(ids)) == (5, 0, 5)


def test_the_relations_are_listed_in_the_order_they_first_appear():
    assert relations_in(FIVE) == ["works_at", "works_on", "based_in"]
