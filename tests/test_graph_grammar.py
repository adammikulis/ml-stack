"""The constraint on a tool call's ids: the schema a turn answers under, the GBNF of it,
and the JSON a constrained reply comes back as."""

from __future__ import annotations

import json

from ml_stack.graph.ask import TERSE, TOOLS, _schema
from ml_stack.graph.grammar import (CAP, ID_FIELDS, call_from, call_schema, constrained,
                                    ids_grammar)

IDS = ["person:ada", "person:bea", 'org:pel"lard']


def _named(schemas):
    return {s["function"]["name"]: s for s in schemas}


def test_constrained_copies_pin_every_id_field_and_leave_the_rest_alone():
    told = _named(constrained(TOOLS, IDS))
    assert told["look_at"]["function"]["parameters"]["properties"]["ids"]["items"] == {
        "enum": IDS}
    assert told["show"]["function"]["parameters"]["properties"]["ids"]["items"] == {"enum": IDS}
    assert told["look_around"]["function"]["parameters"]["properties"]["ids"]["items"] == {
        "enum": IDS}
    between = told["path_between"]["function"]["parameters"]["properties"]
    assert between["from_id"]["enum"] == IDS and between["to_id"]["enum"] == IDS
    assert "type" not in between["from_id"] and between["from_id"]["description"]
    # what does not take an id is the same object, byte for byte
    assert told["look_up"] == _schema("look_up") and told["list_kind"] == _schema("list_kind")
    # the originals were not written on
    assert _schema("look_at")["function"]["parameters"]["properties"]["ids"]["items"] == {
        "type": "string"}
    assert set(ID_FIELDS) == {"look_at", "look_around", "show", "path_between"}


def test_call_schema_is_one_tool_call_or_an_answer():
    schema = call_schema(TOOLS, IDS)
    calls = schema["anyOf"]
    assert [c["properties"]["name"]["const"] for c in calls[:-1]] == [
        s["function"]["name"] for s in TOOLS]
    assert all(c["required"] == ["name", "arguments"] for c in calls[:-1])
    assert calls[-1] == {"type": "object", "properties": {"answer": {"type": "string"}},
                         "required": ["answer"]}
    look_at = calls[1]["properties"]["arguments"]
    assert look_at["properties"]["ids"]["items"] == {"enum": IDS}
    # the terse set constrains the same way
    assert call_schema(TERSE, IDS)["anyOf"][1]["properties"]["arguments"]["properties"][
        "ids"]["items"] == {"enum": IDS}


def test_call_schema_is_none_when_nothing_offered_takes_an_id_or_over_the_cap():
    assert call_schema([_schema("list_kind"), _schema("look_up")], IDS) is None
    assert call_schema(TOOLS, []) is None
    assert call_schema(TOOLS, [f"person:{n}" for n in range(CAP + 1)]) is None
    assert call_schema(TOOLS, [f"person:{n}" for n in range(CAP)]) is not None


def test_ids_grammar_lists_the_ids_as_literals_escaped_for_gbnf():
    text = ids_grammar(IDS)
    assert text.startswith("root ::= ")
    for one in ("person:ada", "person:bea"):
        assert f'"\\"{one}\\""' in text
    # a quote inside an id is escaped twice: once for JSON, once for GBNF
    assert '"\\"org:pel\\\\\\"lard\\""' in text
    for name in ("look_at", "show", "path_between", "look_up", "list_kind"):
        assert f'"\\"{name}\\""' in text
    assert '"\\"answer\\""' in text
    # an invented id is not in it anywhere
    assert "person:ghost" not in text
    # every rule referred to is defined
    defined = {line.split(" ::= ")[0] for line in text.splitlines()}
    assert "ws" in defined and "string" in defined and "integer" in defined
    assert text == ids_grammar(iter(IDS)), "any iterable of ids"


def test_ids_grammar_is_empty_over_the_cap_and_with_nothing_to_constrain():
    assert ids_grammar([f"person:{n}" for n in range(CAP + 1)]) == ""
    assert ids_grammar([]) == ""
    assert ids_grammar(IDS, [_schema("list_kind")]) == ""


def test_call_from_reads_a_tool_call_or_an_answer_out_of_a_constrained_reply():
    calls, answer = call_from(json.dumps({"name": "look_at", "arguments": {"ids": IDS[:1]}}))
    assert answer is None
    assert calls == [{"id": "c1", "type": "function", "function": {
        "name": "look_at", "arguments": json.dumps({"ids": IDS[:1]})}}]
    assert call_from('{"answer": "Ada and Bea both work on compilers."}') == (
        None, "Ada and Bea both work on compilers.")
    # the fence some templates wrap the JSON in is not part of the answer
    assert call_from('```json\n{"answer": "plain"}\n```') == (None, "plain")
    assert call_from("Ada and Bea both work on compilers.") == (None, None)
    assert call_from("") == (None, None)
    assert call_from('{"name": 3}') == (None, None)
    assert call_from('{"name": "show"}') == ([{"id": "c1", "type": "function", "function": {
        "name": "show", "arguments": "{}"}}], None), "arguments left out are none"
