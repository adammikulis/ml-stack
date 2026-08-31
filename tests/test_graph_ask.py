"""Asking a model a question about a graph.

The model is a stand-in that records what it was given and replays a script of tool calls, so
the tools themselves — which are the part with judgement in them — run for real against a real
graph. What is asserted is what the tools returned and what came back as touched.
"""

from dataclasses import dataclass
from typing import Any

from ml_stack.graph.ask import (Answer, converse, converse_stream, look_at, look_up,
                                path_between, tools_for)

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
    thinking: str | None = None


class ScriptedModel:
    """Answers with the tool calls it was told to, then with words."""

    def __init__(self, script):
        self.script = list(script)
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None, **_):
        self.seen.append(list(messages))
        # with no tools offered there is nothing to call, so it answers in words
        if tools and self.script:
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


def test_a_tool_the_caller_adds_is_offered_and_called():
    seen = {}

    def census(args):
        seen["args"] = dict(args)
        return {"people": 2}

    schema = {"type": "function", "function": {
        "name": "head_count",
        "description": "How many entries of a kind the graph holds.",
        "parameters": {"type": "object", "properties": {"kind": {"type": "string"}},
                       "required": ["kind"]}}}
    model = ScriptedModel([call("head_count", kind="person")])
    out = converse("how many people?", GRAPH, model,
                   tools=[*tools_for(GRAPH), (schema, census)])
    assert seen["args"] == {"kind": "person"}
    assert out.steps == ["used head_count"]
    tool_turns = [m for turn in model.seen for m in turn if m.get("role") == "tool"]
    assert any('"people": 2' in m["content"] for m in tool_turns)


def test_what_was_read_is_told_apart_from_what_was_merely_found():
    model = ScriptedModel([call("look_up", text="compil"),
                           call("look_at", ids=["person:bea"])])
    out = converse("what does Bea do?", GRAPH, model, limit=2)
    assert out.found == ["topic:compilers", "person:ada", "person:bea"]
    assert out.read == ["person:bea"]
    assert out.path == []
    # the cap keeps what was read; only what was merely found falls off
    assert out.ids == ["person:bea", "topic:compilers"]


def test_a_question_needing_no_tools_still_answers():
    out = converse("hello", GRAPH, ScriptedModel([]))
    assert out.ids == [] and out.steps == [] and out.content


def test_running_out_of_rounds_still_answers():
    """The last reply of an exhausted loop is a tool call; the question still deserves words."""
    asking = [call("look_up", text="Ada Lovelace")] * 4
    model = ScriptedModel(asking)
    out = converse("who?", GRAPH, model, rounds=2)
    assert out.content == "Ada and Bea both work on compilers."
    assert out.ids == ["person:ada"]
    # the last call was made with no tools offered, which is what let it answer
    assert len(model.seen) == 3


def test_held_entries_are_named_to_the_model_by_label_and_id():
    model = ScriptedModel([])
    converse("and what else?", GRAPH, model, held=["person:ada", "topic:compilers"])
    system = model.seen[0][0]["content"]
    assert "Currently highlighted: Ada Lovelace (person:ada), compilers (topic:compilers)" in system


def test_a_follow_up_that_re_reads_held_keeps_it_in_the_answer():
    model = ScriptedModel([call("look_at", ids=["person:ada"]),
                           call("look_up", text="Bea Marlow")])
    out = converse("also show Bea", GRAPH, model, held=["person:ada"])
    assert out.ids == ["person:ada", "person:bea"]


def test_a_subject_change_does_not_drag_held_along():
    model = ScriptedModel([call("look_up", text="Pellard")])
    out = converse("what is Pellard Foundry?", GRAPH, model, held=["person:ada", "person:bea"])
    assert out.ids == ["org:pellard"]


def test_an_unknown_held_id_is_dropped_silently():
    model = ScriptedModel([])
    out = converse("hello", GRAPH, model, held=["person:ghost"])
    assert out.ids == []
    assert "Currently highlighted" not in model.seen[0][0]["content"]


def test_a_model_that_goes_quiet_is_told_to_answer():
    """Rounds exhausted, the no-tools re-ask comes back empty: one plain nudge gets words."""

    class Quiet(ScriptedModel):
        def __init__(self, script):
            super().__init__(script)
            self.silences = 1

        def chat(self, messages, tools=None, **kw):
            if not tools and self.silences:
                self.silences -= 1
                self.seen.append(list(messages))
                return Reply(content="")
            return super().chat(messages, tools=tools, **kw)

    model = Quiet([("look_up", {"text": "compilers"})] * 5)
    out = converse("who works on compilers?", GRAPH, model, rounds=5)
    assert out.content
    assert "plain words" in model.seen[-1][-1]["content"]


def test_a_model_that_stops_calling_tools_without_answering_is_nudged():
    """It ran its tools, then returned an empty message rather than an answer."""

    class Silent(ScriptedModel):
        def __init__(self, script):
            super().__init__(script)
            self.quiet = True

        def chat(self, messages, tools=None, **kw):
            self.seen.append(list(messages))
            if self.script and tools:
                import json
                name, args = self.script.pop(0)
                return Reply(tool_calls=[{"id": "c1", "function": {
                    "name": name, "arguments": json.dumps(args)}}])
            if self.quiet:
                self.quiet = False
                return Reply(content="   ")
            return Reply(content="Ada works on compilers.")

    model = Silent([("look_up", {"text": "compilers"})])
    out = converse("who works on compilers?", GRAPH, model, rounds=5)
    assert out.content == "Ada works on compilers."
    assert "plain words" in model.seen[-1][-1]["content"]


def test_an_answer_that_arrives_only_as_thinking_is_not_lost():
    """gpt-oss can put every word in the reasoning channel and none in content."""

    class Reasoner(ScriptedModel):
        def chat(self, messages, tools=None, **kw):
            self.seen.append(list(messages))
            if tools and self.script:
                import json
                name, args = self.script.pop(0)
                return Reply(tool_calls=[{"id": "c1", "function": {
                    "name": name, "arguments": json.dumps(args)}}])
            return Reply(content="", thinking="Ada is the one working on compilers.")

    model = Reasoner([call("look_at", ids=["person:ada"])])
    out = converse("who works on compilers?", GRAPH, model)
    assert out.content == "Ada is the one working on compilers."
    assert "plain words" in model.seen[-1][-1]["content"]


def test_a_model_that_only_searched_gets_the_top_finds_read_to_it():
    """All look_ups and no look_at: the finds are read out before the final ask."""

    class Searcher(ScriptedModel):
        def chat(self, messages, tools=None, **kw):
            self.seen.append(list(messages))
            if tools and self.script:
                import json
                name, args = self.script.pop(0)
                return Reply(tool_calls=[{"id": "c1", "function": {
                    "name": name, "arguments": json.dumps(args)}}])
            if any("What the graph holds" in str(m.get("content") or "")
                   for m in messages if m.get("role") == "user"):
                return Reply(content="Ada and Bea, going by what they said.")
            return Reply(content="")

    model = Searcher([call("look_up", text="compilers")] * 3)
    out = converse("who works on compilers?", GRAPH, model, rounds=3)
    assert out.content == "Ada and Bea, going by what they said."
    assert out.read == ["topic:compilers", "person:ada", "person:bea"]
    assert out.steps[-1] == "read the top 3 finds"
    handed = model.seen[-1][-1]["content"]
    assert "Ada Lovelace (person)" in handed and "plain words" in handed


def test_streaming_reports_tools_and_the_answer_in_order():
    events = []
    model = ScriptedModel([call("look_up", text="Ada Lovelace"),
                           call("look_at", ids=["person:ada"])])
    out = converse_stream("who is Ada?", GRAPH, model, on_event=events.append)
    assert [e["event"] for e in events] \
        == ["tool", "tool_result", "tool", "tool_result", "answer", "done"]
    assert events[0] == {"event": "tool", "name": "look_up", "detail": "'Ada Lovelace'"}
    assert events[1] == {"event": "tool_result", "name": "look_up", "count": 1}
    assert events[2]["detail"] == "1 id"
    assert events[4]["text"] == out.content == "Ada and Bea both work on compilers."
    assert out.ids == ["person:ada"]


def test_streaming_passes_deltas_through_as_they_arrive():
    """A client whose chat takes on_delta streams thinking and answer piece by piece."""

    class Streamer:
        def chat(self, messages, tools=None, on_delta=None, **kw):
            for piece in ("weigh ", "it up"):
                on_delta("thinking", piece)
            for piece in ("Ada works ", "on compilers."):
                on_delta("content", piece)
            return Reply(content="Ada works on compilers.", thinking="weigh it up")

    events = []
    out = converse_stream("who?", GRAPH, Streamer(), on_event=events.append)
    assert [e["event"] for e in events] \
        == ["thinking", "thinking", "answer", "answer", "done"]
    assert [e.get("text") for e in events[:2]] == ["weigh ", "it up"]
    assert [e.get("text") for e in events[2:4]] == ["Ada works ", "on compilers."]
    assert out.content == "Ada works on compilers."


def test_a_client_without_deltas_still_streams_whole_pieces():
    """The scripted client ignores on_delta; its thinking and answer arrive as one event
    each."""

    class Muser(ScriptedModel):
        def chat(self, messages, tools=None, **kw):
            self.seen.append(list(messages))
            return Reply(content="Nobody here.", thinking="an empty room")

    events = []
    out = converse_stream("who?", GRAPH, Muser([]), on_event=events.append)
    assert [e["event"] for e in events] == ["thinking", "answer", "done"]
    assert events[0]["text"] == "an empty room"
    assert events[1]["text"] == "Nobody here."
    assert out.content == "Nobody here."
