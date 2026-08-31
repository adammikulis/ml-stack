"""A change request that reaches past the claimant's own information is flagged."""

from ml_stack.graph.concerns import concerns

GRAPH = {
    "nodes": [{"id": "p:ada", "kind": "person", "label": "Ada Lovelace", "attrs": {}},
              {"id": "p:bea", "kind": "person", "label": "Bea Marlow", "attrs": {}},
              {"id": "o:foundry", "kind": "org", "label": "Pellard Foundry", "attrs": {}}],
    "edges": [{"source": "p:ada", "rel": "works_at", "target": "o:foundry"},
              {"source": "p:bea", "rel": "works_at", "target": "o:foundry"}],
}


def test_a_request_about_your_own_edge_raises_nothing():
    req = {"attested": True, "claimed": "p:ada"}
    edits = [{"op": "remove_relation", "target": "p:ada|works_at|o:foundry"}]
    assert concerns(req, edits, GRAPH) == []


def test_asking_to_change_someone_else_is_flagged():
    req = {"attested": True, "claimed": "p:ada"}
    edits = [{"op": "remove_relation", "target": "p:bea|works_at|o:foundry"}]
    assert any("not theirs" in c for c in concerns(req, edits, GRAPH))


def test_an_unattested_request_is_flagged():
    req = {"attested": False, "claimed": "p:ada"}
    assert any("own information" in c for c in concerns(req, [], GRAPH))


def test_claiming_to_be_nobody_in_the_graph_is_flagged():
    assert any("not in the graph" in c
               for c in concerns({"attested": True, "claimed": "p:nobody"}, [], GRAPH))


def test_not_saying_who_you_are_is_flagged():
    assert any("did not say who" in c for c in concerns({"attested": True}, [], GRAPH))


def test_removing_a_node_others_are_joined_to_is_flagged():
    req = {"attested": True, "claimed": "p:ada"}
    edits = [{"op": "remove", "target": "o:foundry"}]
    assert any("shared with" in c for c in concerns(req, edits, GRAPH))


def test_a_node_only_the_claimant_touches_raises_nothing():
    graph = {"nodes": [{"id": "p:ada", "kind": "person", "label": "Ada Lovelace", "attrs": {}},
                       {"id": "t:looms", "kind": "topic", "label": "looms", "attrs": {}}],
             "edges": [{"source": "p:ada", "rel": "interested_in", "target": "t:looms"}]}
    req = {"attested": True, "claimed": "p:ada"}
    assert concerns(req, [{"op": "remove", "target": "t:looms"}], graph) == []
