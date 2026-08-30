"""Graph edits: a model may pick from the listing, and may not invent anything."""

from __future__ import annotations

import json

import pytest
from conftest import json_reply
from ml_stack.client import Client
from ml_stack.contracts import ContractError, grammar_for
from ml_stack.entities import (EDITS_SCHEMA, OPERATIONS, Edit, ids_of, objections, plan_edits,
                               validate_edits)

NODES = ["n:1", "n:2", "n:3"]
EDGES = ["e:1"]


def keep(*raw):
    return validate_edits({"edits": list(raw)}, node_ids=NODES, edge_ids=EDGES)


def test_a_well_formed_edit_is_kept():
    assert keep({"op": "rename", "target": "n:1", "name": "", "value": "Ada",
                 "reason": "call it Ada"}) == [
        Edit(op="rename", target="n:1", value="Ada", reason="call it Ada")]


def test_an_unknown_op_is_dropped():
    assert keep({"op": "delete_everything", "target": "n:1", "value": "", "reason": "go on"}) == []
    assert keep({"op": "", "target": "n:1", "value": "x", "reason": ""}) == []
    assert keep({"op": 7, "target": "n:1", "value": "x", "reason": ""}) == []


def test_an_unknown_target_is_dropped():
    assert keep({"op": "remove", "target": "n:99", "value": "", "reason": "gone"}) == []
    assert keep({"op": "remove", "target": "", "value": "", "reason": "gone"}) == []
    assert keep({"op": "remove", "value": "", "reason": "gone"}) == []


def test_empty_and_garbage_input_return_nothing():
    for raw in ({}, {"edits": []}, {"edits": None}, {"edits": "n:1"}, [], None, "", "n:1", 3,
                {"edits": [None, 5, "n:1", []]}, {"other": [{"op": "remove", "target": "n:1"}]}):
        assert validate_edits(raw, node_ids=NODES, edge_ids=EDGES) == []


def test_a_bare_list_of_edits_is_read_too():
    assert validate_edits([{"op": "remove", "target": "n:2", "reason": "stale"}],
                          node_ids=NODES) == [Edit(op="remove", target="n:2", reason="stale")]


def test_merge_needs_a_second_node_that_is_not_the_first():
    assert keep({"op": "merge", "target": "n:1", "value": "n:2", "reason": "same thing"}) == [
        Edit(op="merge", target="n:1", value="n:2", reason="same thing")]
    assert keep({"op": "merge", "target": "n:1", "value": "n:1", "reason": "same thing"}) == []
    assert keep({"op": "merge", "target": "n:1", "value": "n:9", "reason": "same thing"}) == []
    assert keep({"op": "merge", "target": "n:1", "value": "Ada", "reason": "same thing"}) == []
    assert keep({"op": "merge", "target": "e:1", "value": "n:2", "reason": "same thing"}) == []


def test_a_relation_joins_two_nodes_under_a_name():
    assert keep({"op": "add_relation", "target": "n:1", "name": "knows", "value": "n:2",
                 "reason": "they met"}) == [
        Edit(op="add_relation", target="n:1", name="knows", value="n:2", reason="they met")]
    assert keep({"op": "add_relation", "target": "n:1", "name": "", "value": "n:2"}) == []
    assert keep({"op": "add_relation", "target": "n:1", "name": "knows", "value": "Ada"}) == []


def test_removing_a_relation_takes_an_edge_id():
    assert keep({"op": "remove_relation", "target": "e:1", "reason": "wrong"}) == [
        Edit(op="remove_relation", target="e:1", reason="wrong")]
    assert keep({"op": "remove_relation", "target": "n:1", "reason": "wrong"}) == []


def test_setting_an_attribute_needs_the_attribute_named():
    assert keep({"op": "set_attr", "target": "e:1", "name": "weight", "value": "3",
                 "reason": "heavier"}) == [
        Edit(op="set_attr", target="e:1", name="weight", value="3", reason="heavier")]
    assert keep({"op": "set_attr", "target": "n:1", "name": "", "value": "3"}) == []
    assert keep({"op": "set_attr", "target": "n:1", "name": "weight", "value": ""}) == []


def test_rename_needs_a_new_label():
    assert keep({"op": "rename", "target": "n:1", "value": "", "reason": "x"}) == []


def test_the_ops_that_take_no_name_or_value_lose_them():
    assert keep({"op": "remove", "target": "n:1", "name": "weight", "value": "n:2",
                 "reason": "gone"}) == [
        Edit(op="remove", target="n:1", reason="gone")]


def test_a_non_string_field_drops_the_edit():
    assert keep({"op": "rename", "target": "n:1", "value": 4, "reason": "x"}) == []
    assert keep({"op": "rename", "target": ["n:1"], "value": "Ada", "reason": "x"}) == []
    assert keep({"op": "rename", "target": "n:1", "value": "Ada", "reason": {"a": 1}}) == []


def test_fields_are_stripped():
    assert keep({"op": " rename ", "target": " n:1 ", "value": " Ada ", "reason": " because "}) == [
        Edit(op="rename", target="n:1", value="Ada", reason="because")]


def test_the_same_edit_twice_is_kept_once():
    edit = {"op": "remove", "target": "n:1", "reason": "gone"}
    assert keep(edit, dict(edit, reason="still gone")) == [
        Edit(op="remove", target="n:1", reason="gone")]


def test_good_edits_survive_the_bad_ones_around_them():
    assert keep({"op": "nope", "target": "n:1"},
                {"op": "remove", "target": "n:1", "reason": "gone"},
                {"op": "remove", "target": "n:404"}) == [
        Edit(op="remove", target="n:1", reason="gone")]


def test_every_op_in_the_closed_set_can_be_produced():
    made = {e.op for e in keep(
        {"op": "rename", "target": "n:1", "value": "Ada", "reason": "r"},
        {"op": "remove", "target": "n:2", "reason": "r"},
        {"op": "merge", "target": "n:3", "value": "n:1", "reason": "r"},
        {"op": "set_attr", "target": "n:1", "name": "kind", "value": "x", "reason": "r"},
        {"op": "add_relation", "target": "n:1", "name": "knows", "value": "n:2", "reason": "r"},
        {"op": "remove_relation", "target": "e:1", "reason": "r"},
    )}
    assert made == set(OPERATIONS)


def test_ids_come_out_of_a_mapping_a_record_list_or_a_string_list():
    assert ids_of({"n:1": {"label": "Ada"}, "n:2": {}}) == ["n:1", "n:2"]
    assert ids_of([{"id": "n:1"}, {"label": "no id"}, {"id": 3}]) == ["n:1"]
    assert ids_of(["n:1", "n:2"]) == ["n:1", "n:2"]
    assert ids_of(None) == [] and ids_of("n:1") == []


def test_objections_name_what_was_dropped():
    said = objections({"edits": [{"op": "nope", "target": "n:1"},
                                 {"op": "remove", "target": "n:404"},
                                 {"op": "set_attr", "target": "n:1", "value": "x"},
                                 {"op": "remove", "target": "n:1"}]},
                      node_ids=NODES, edge_ids=EDGES)
    assert len(said) == 3
    assert "'nope'" in said[0] and "'n:404'" in said[1] and "set_attr" in said[2]


def test_objections_on_a_reply_with_no_edits():
    assert objections({}, node_ids=NODES) == ["the reply has no edits array"]
    assert objections({"edits": []}, node_ids=NODES) == []


class FakeClient:
    """A Client.extract stand-in that replays one reply and records the call."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def extract(self, text, schema, **kwargs):
        self.calls.append((text, schema, kwargs))
        return self.reply


def test_plan_edits_keeps_only_what_checks_out():
    client = FakeClient({"edits": [
        {"op": "rename", "target": "n:1", "name": "", "value": "Ada", "reason": "call it Ada"},
        {"op": "rename", "target": "n:404", "name": "", "value": "Ghost", "reason": "invented"},
        {"op": "obliterate", "target": "n:1", "name": "", "value": "", "reason": "invented"},
    ]})
    out = plan_edits("call node one Ada", nodes={"n:1": {"label": "one"}}, edges=[], client=client)

    assert out == [Edit(op="rename", target="n:1", value="Ada", reason="call it Ada")]
    text, schema, kwargs = client.calls[0]
    assert "n:1: one" in text and "call node one Ada" in text
    assert schema is EDITS_SCHEMA
    assert kwargs["think"] is False
    assert kwargs["check"]({"edits": [{"op": "rename", "target": "n:9", "value": "x"}]})


def test_plan_edits_asks_nothing_when_there_is_no_request_or_no_graph():
    client = FakeClient({"edits": []})
    assert plan_edits("   ", nodes=NODES, client=client) == []
    assert plan_edits("rename something", nodes=[], edges=[], client=client) == []
    assert client.calls == []


def test_plan_edits_survives_a_reply_that_is_not_a_document():
    assert plan_edits("do it", nodes=NODES, client=FakeClient("sorry, no")) == []
    assert plan_edits("do it", nodes=NODES, client=FakeClient(None)) == []


def test_the_schema_becomes_a_grammar():
    grammar = grammar_for(EDITS_SCHEMA)
    assert "root ::=" in grammar
    for op in OPERATIONS:
        assert f'"{op}"' in grammar.replace('\\"', '"')
    assert grammar_for(EDITS_SCHEMA) == grammar


def test_the_schema_holds_nothing_grammar_for_refuses():
    with pytest.raises(ContractError):
        grammar_for(dict(EDITS_SCHEMA, type="tuple"))


def test_plan_edits_through_a_client_against_a_real_server(server):
    reply = {"edits": [{"op": "merge", "target": "n:1", "name": "", "value": "n:2",
                        "reason": "one and two are the same"}]}
    seen = []

    def handler(method, path, body):
        seen.append((path, json.loads(body)))
        return json_reply({"choices": [{"message": {"role": "assistant",
                                                    "content": json.dumps(reply)}}]})

    instance = server(handler)
    out = plan_edits("one and two are the same node",
                     nodes=[{"id": "n:1", "label": "one"}, {"id": "n:2", "label": "two"}],
                     client=Client(instance.base_url))

    assert out == [Edit(op="merge", target="n:1", value="n:2",
                        reason="one and two are the same")]
    path, sent = seen[0]
    assert path == "/v1/chat/completions"
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["response_format"]["json_schema"]["name"] == "graph_edits"
    assert "n:2: two" in sent["messages"][1]["content"]
