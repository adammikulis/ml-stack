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
    out = concerns(req, edits, GRAPH)
    assert any("shared with" in c for c in out)
    assert any("their own link" in c for c in out)


def test_a_node_only_the_claimant_touches_raises_nothing():
    graph = {"nodes": [{"id": "p:ada", "kind": "person", "label": "Ada Lovelace", "attrs": {}},
                       {"id": "t:looms", "kind": "topic", "label": "looms", "attrs": {}}],
             "edges": [{"source": "p:ada", "rel": "interested_in", "target": "t:looms"}]}
    req = {"attested": True, "claimed": "p:ada"}
    assert concerns(req, [{"op": "remove", "target": "t:looms"}], graph) == []


FAMILY = {
    "nodes": [{"id": "p:ada", "kind": "person", "label": "Ada Lovelace", "attrs": {}},
              {"id": "p:bea", "kind": "person", "label": "Bea Marlow", "attrs": {}},
              {"id": "p:ada son", "kind": "person", "label": "Ada Lovelace's son", "attrs": {}}],
    "edges": [{"source": "p:ada", "rel": "parent_of", "target": "p:ada son"}],
}


def test_removing_a_satellite_joined_only_to_the_claimant_raises_nothing():
    req = {"attested": True, "claimed": "p:ada"}
    assert concerns(req, [{"op": "remove", "target": "p:ada son"}], FAMILY) == []


def test_removing_a_satellite_another_member_is_joined_to_points_at_the_edge():
    graph = {"nodes": list(FAMILY["nodes"]),
             "edges": [{"source": "p:ada", "rel": "parent_of", "target": "p:ada son"},
                       {"source": "p:bea", "rel": "mentors", "target": "p:ada son"}]}
    req = {"attested": True, "claimed": "p:ada"}
    out = concerns(req, [{"op": "remove", "target": "p:ada son"}], graph)
    assert any("shared with 1" in c for c in out)
    assert any("their own link" in c for c in out)


def test_removing_your_own_node_while_others_are_joined_to_it_is_flagged():
    req = {"attested": True, "claimed": "p:ada"}
    out = concerns(req, [{"op": "remove", "target": "p:ada"}], FAMILY)
    assert any("shared with" in c and "their own link" in c for c in out)


def test_removing_your_own_node_that_nothing_is_joined_to_raises_nothing():
    graph = {"nodes": [{"id": "p:ada", "kind": "person", "label": "Ada Lovelace", "attrs": {}}],
             "edges": []}
    req = {"attested": True, "claimed": "p:ada"}
    assert concerns(req, [{"op": "remove", "target": "p:ada"}], graph) == []


def test_an_edge_edit_naming_its_ends_by_label_is_still_the_claimants_own():
    req = {"attested": True, "claimed": "p:ada"}
    edits = [{"op": "remove_edge", "target": "Ada Lovelace", "other": "Pellard Foundry",
              "name": "works_at"}]
    assert concerns(req, edits, GRAPH) == []


def test_an_edge_edit_by_label_not_touching_the_claimant_is_flagged():
    req = {"attested": True, "claimed": "p:ada"}
    edits = [{"op": "remove_edge", "target": "Bea Marlow", "other": "Pellard Foundry",
              "name": "works_at"}]
    assert any("not theirs" in c for c in concerns(req, edits, GRAPH))
