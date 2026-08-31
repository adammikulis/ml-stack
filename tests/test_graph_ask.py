"""Asking a model a question about a graph.

The model is a stand-in that records what it was given and replays a script of tool calls, so
the tools themselves — which are the part with judgement in them — run for real against a real
graph. What is asserted is what the tools returned and what came back as touched.
"""

from dataclasses import dataclass
from typing import Any

from ml_stack.graph.ask import Answer, converse, look_at, look_up, path_between

GRAPH = {
    "nodes": [
        {"id": "person:ada", "kind": "person", "label": "Ada Lovelace", "mentions": 4,
         "attrs": {"role": "analyst", "location": "Turin"}, "messages": ["m1"]},
        {"id": "person:bea", "kind": "person", "label": "Bea Marlow", "mentions": 2,
         "attrs": {}, "messages": ["m2"]},
        {"id": "topic:compilers", "kind": "topic", "label": "compilers", "mentions": 3,
         "attrs": {}, "messages": ["m1", "m2"]},
        {"id": "org:pellard", "kind": "org", "label": "Pellard Foundry", "mentions": 1,
         "attrs": {"type": "company"}, "messages": []},
    ],
    "edges": [
        {"source": "person:ada", "target": "topic:compilers", "rel": "interested_in", "weight": 3},
        {"source": "person:bea", "target": "topic:compilers", "rel": "interested_in", "weight": 2},
        {"source": "person:ada", "target": "org:pellard", "rel": "works_at", "weight": 1},
    ],
    "messages": {
        "m1": {"text": "I am Ada and I have spent years on compilers."},
        "m2": {"text": "compilers are what I do too, mostly."},
    },
}


@dataclass
class Reply:
    content: str = ""
    tool_calls: list | None = None


class ScriptedModel:
    """Answers with the tool calls it was told to, then with words."""

    def __init__(self, script):
        self.script = list(script)
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None, **_):
        self.seen.append(list(messages))
        if self.script:
            name, args = self.script.pop(0)
            import json
            return Reply(tool_calls=[{"id": "c1", "function": {
                "name": name, "arguments": json.dumps(args)}}])
        return Reply(content="Ada and Bea both work on compilers.")


def call(name, **args):
    return (name, args)


def test_look_up_finds_by_name_then_by_attribute_then_by_what_was_said():
    assert [r["id"] for r in look_up(GRAPH, "Ada Lovelace")] == ["person:ada"]
    assert [r["id"] for r in look_up(GRAPH, "compil")][0] == "topic:compilers"
    assert [r["id"] for r in look_up(GRAPH, "Turin")] == ["person:ada"]
    # both the topic and the person carry that message, and both are honest answers
    assert {r["id"] for r in look_up(GRAPH, "mostly")} == {"person:bea", "topic:compilers"}
    assert look_up(GRAPH, "  ") == []
    assert look_up(GRAPH, "nothing here") == []


def test_look_at_reads_out_what_is_held_including_what_was_said():
    text = look_at(GRAPH, ["person:ada", "person:nobody"])
    assert "Ada Lovelace (person)" in text and "analyst" in text and "Turin" in text
    assert "interested_in compilers" in text and "works_at Pellard Foundry" in text
    assert 'said: "I am Ada' in text
    assert look_at(GRAPH, []) == ""


def test_path_between_reads_as_a_chain():
    out = path_between(GRAPH, "person:ada", "person:bea")
    assert out["path"] == ["person:ada", "topic:compilers", "person:bea"]
    assert out["reads"] == "Ada Lovelace → compilers → Bea Marlow"
    assert path_between(GRAPH, "person:ada", "org:nowhere")["path"] == []


def test_what_the_model_touched_is_what_comes_back():
    model = ScriptedModel([call("look_up", text="Ada Lovelace"),
                           call("path_between", from_id="person:ada", to_id="person:bea")])
    out = converse("how are they connected?", GRAPH, model)
    assert isinstance(out, Answer)
    assert out.ids == ["person:ada", "topic:compilers", "person:bea"]
    assert out.steps == ["looked up 'Ada Lovelace'", "traced a path"]
    assert out.content == "Ada and Bea both work on compilers."
    # the tool's answer was actually put in front of the model
    tool_turns = [m for turn in model.seen for m in turn if m.get("role") == "tool"]
    assert any("topic:compilers" in m["content"] for m in tool_turns)


def test_an_id_the_model_invents_is_not_lit_up():
    model = ScriptedModel([call("look_at", ids=["person:ada", "person:ghost"])])
    out = converse("who?", GRAPH, model)
    assert out.ids == ["person:ada"]


def test_a_question_needing_no_tools_still_answers():
    out = converse("hello", GRAPH, ScriptedModel([]))
    assert out.ids == [] and out.steps == [] and out.content
