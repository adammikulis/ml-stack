"""Letting a model propose changes to a graph, without letting it make them.

Every test asserts the graph is unchanged afterwards, because that is the guarantee: these
tools return proposals and something else decides.
"""

import copy
import json

from ml_stack.graph.propose import Change, check, proposing, tools_for

GRAPH = {
    "nodes": [
        {"id": "p:ada", "kind": "person", "label": "Ada Lovelace", "attrs": {}},
        {"id": "p:bea", "kind": "person", "label": "Bea Marlow", "attrs": {}},
        {"id": "t:compilers", "kind": "topic", "label": "compilers", "attrs": {}},
    ],
    "edges": [{"source": "p:ada", "target": "t:compilers", "rel": "interested_in", "weight": 2}],
}


def call(tool, /, **args):
    return {"function": {"name": tool, "arguments": json.dumps(args)}}


def gathered(*calls, graph=GRAPH):
    before = copy.deepcopy(graph)
    _, gather = proposing(graph)
    out = gather(list(calls))
    assert graph == before, "a proposal changed the graph"
    return out


def test_the_tools_describe_this_graphs_own_vocabulary():
    tools = tools_for(GRAPH)
    assert [t["function"]["name"] for t in tools] == [
        "add_node", "add_edge", "rename", "set_attribute", "unset_attribute", "remove_node",
        "remove_edge", "merge_nodes"]
    kinds = tools[0]["function"]["parameters"]["properties"]["kind"]["description"]
    assert "person, topic" in kinds
    rels = tools[1]["function"]["parameters"]["properties"]["rel"]["description"]
    assert "interested_in" in rels


def test_a_sound_proposal_comes_back_sound_and_unapplied():
    out = gathered(call("add_edge", from_id="p:bea", to_id="t:compilers", rel="interested_in",
                        reason="she said compilers are what she does"))
    assert len(out) == 1 and out[0].sound
    assert out[0].describe().startswith("join p:bea -interested_in-> t:compilers")


def test_a_proposal_about_something_that_is_not_there_says_so():
    out = gathered(call("remove_node", id="p:nobody", reason="asked"),
                   call("add_edge", from_id="p:ada", to_id="t:nothing", rel="knows", reason="x"))
    assert [c.sound for c in out] == [False, False]
    assert "is not in the graph" in out[0].problems[0]
    assert "the second entry" in out[1].problems[0]


def test_proposing_what_the_graph_already_has_is_a_duplicate():
    out = gathered(call("add_node", label="Ada Lovelace", kind="person", reason="new person"),
                   call("add_edge", from_id="p:ada", to_id="t:compilers", rel="interested_in",
                        reason="again"),
                   call("remove_edge", from_id="p:bea", to_id="t:compilers", rel="interested_in",
                        reason="not hers"))
    assert [c.sound for c in out] == [False, False, False]
    assert "already in the graph" in out[0].problems[0]
    assert "already joined" in out[1].problems[0]
    assert "not joined" in out[2].problems[0]


def test_folding_two_different_kinds_of_thing_is_refused():
    out = gathered(call("merge_nodes", keep_id="p:ada", remove_id="t:compilers", reason="same"),
                   call("merge_nodes", keep_id="p:ada", remove_id="p:ada", reason="same"),
                   call("merge_nodes", keep_id="p:ada", remove_id="p:bea", reason="same person"))
    assert [c.sound for c in out] == [False, False, True]
    assert "different kinds" in out[0].problems[0]
    assert "same entry" in out[1].problems[0]


def test_a_change_with_no_reason_is_not_sound():
    out = gathered(call("rename", id="p:ada", label="Ada L", reason=""))
    assert not out[0].sound and out[0].problems == ["no reason given for rename p:ada"]


def test_a_tool_nobody_offered_is_ignored_and_bad_arguments_are_survived():
    _, gather = proposing(GRAPH)
    assert gather([{"function": {"name": "drop_database", "arguments": "{}"}}]) == []
    out = gather([{"function": {"name": "rename", "arguments": "not json at all"}}])
    assert len(out) == 1 and not out[0].sound


def test_an_empty_graph_still_offers_the_tools():
    tools, gather = proposing({"nodes": [], "edges": []})
    assert len(tools) == 8
    out = gather([call("add_node", label="Ada", kind="person", reason="first entry")])
    assert out[0].sound


def test_check_can_be_used_on_its_own():
    made = check(GRAPH, Change(op="add_node", value="Cyd Marek", name="person", reason="new"))
    assert made.sound


def test_an_attribute_can_be_proposed_unset_without_a_value():
    tools = tools_for(GRAPH)
    unset = next(t["function"] for t in tools if t["function"]["name"] == "unset_attribute")
    assert unset["parameters"]["required"] == ["id", "name", "reason"]
    out = gathered(call("unset_attribute", id="p:ada", name="role", reason="she left the role"),
                   call("unset_attribute", id="p:nobody", name="role", reason="gone"),
                   call("unset_attribute", id="p:ada", name="", reason="gone"))
    assert [c.sound for c in out] == [True, False, False]
    assert out[0].describe().startswith("unset role of p:ada")
    assert "is not in the graph" in out[1].problems[0]
    assert "no attribute named" in out[2].problems[0]


def test_a_change_with_no_reason_names_the_op_and_the_ids():
    out = gathered(call("remove_edge", from_id="p:ada", to_id="t:compilers", rel="interested_in",
                        reason=""))
    assert not out[0].sound
    assert out[0].problems == ["no reason given for remove_edge p:ada -interested_in-> t:compilers"]
