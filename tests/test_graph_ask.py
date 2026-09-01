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
        # A model can only call what it was offered. The last turn offers the tools that act
        # but not the ones that search, so a script that wants to search then has nothing to
        # call and answers in words — which is the whole point of taking them away.
        offered = {str((t.get("function") or {}).get("name")) for t in (tools or [])}
        if self.script and self.script[0][0] in offered:
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
    # the last call offers no way to keep searching, which is what let it answer
    # two rounds of tools, the ask that follows, and the ask for what to light
    assert len(model.seen) == 4


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
            # the answering turn is the one with no way left to search
            searching = {"look_up", "look_at", "path_between"}
            offered = {str((t.get("function") or {}).get("name")) for t in (tools or [])}
            if not (offered & searching) and self.silences:
                self.silences -= 1
                self.seen.append(list(messages))
                return Reply(content="")
            return super().chat(messages, tools=tools, **kw)

    model = Quiet([("look_up", {"text": "compilers"})] * 5)
    out = converse("who works on compilers?", GRAPH, model, rounds=5)
    assert out.content
    assert any("plain words" in m["content"] for said in model.seen for m in said)


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
    assert any("plain words" in m["content"] for said in model.seen for m in said)


SCRATCH = "Actually the look_at shows Ada; need to check Bea. Wait \u2014 maybe compilers?"


def test_an_answer_left_in_the_thinking_channel_is_asked_for_again():
    """gpt-oss can put everything in the reasoning channel and none in content."""

    class Reasoner(ScriptedModel):
        def __init__(self, script):
            super().__init__(script)
            self.reasoned = False

        def chat(self, messages, tools=None, **kw):
            self.seen.append(list(messages))
            if tools and self.script:
                import json
                name, args = self.script.pop(0)
                return Reply(tool_calls=[{"id": "c1", "function": {
                    "name": name, "arguments": json.dumps(args)}}])
            if not self.reasoned:
                self.reasoned = True
                return Reply(content="", thinking=SCRATCH)
            return Reply(content="Ada works on compilers.")

    model = Reasoner([call("look_at", ids=["person:ada"])])
    out = converse("who works on compilers?", GRAPH, model)
    assert out.content == "Ada works on compilers."
    # recovered by whichever ask reached it first — the plain nudge or the one that offers
    # the notes back; both are asking for the answer rather than printing the working
    said = [m["content"] for turn in model.seen for m in turn]
    assert any("plain words" in x or "working towards" in x for x in said)


def test_the_working_out_is_never_shown_as_the_answer():
    """A scratchpad reads as a broken machine: say plainly that no answer came."""

    class OnlyThinks(ScriptedModel):
        def chat(self, messages, tools=None, **kw):
            self.seen.append(list(messages))
            if tools and self.script:
                import json
                name, args = self.script.pop(0)
                return Reply(tool_calls=[{"id": "c1", "function": {
                    "name": name, "arguments": json.dumps(args)}}])
            return Reply(content="", thinking=SCRATCH)

    out = converse("who works on compilers?", GRAPH, OnlyThinks([call("look_at", ids=["person:ada"])]))
    assert SCRATCH not in out.content
    assert "did not finish an answer" in out.content
    assert out.read == ["person:ada"]


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
    handed = [m["content"] for turn in model.seen for m in turn]
    assert any("Ada Lovelace (person)" in x and "plain words" in x for x in handed)


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
    # it answered without searching, so it is sent back to look once and streams again; the
    # reader is told to start over rather than have the second answer appended to the first
    assert [e["event"] for e in events] \
        == ["thinking", "thinking", "answer", "answer",
            "restart", "thinking", "thinking", "answer", "answer", "done"]
    assert [e.get("text") for e in events[:2]] == ["weigh ", "it up"]
    assert [e.get("text") for e in events[2:4]] == ["Ada works ", "on compilers."]
    assert out.content == "Ada works on compilers."


def test_a_streamed_answer_that_did_search_is_not_restarted():
    """The restart is for an answer built on nothing, not for every answer."""

    model = ScriptedModel([call("look_at", ids=["person:ada"]),
                           call("show", ids=["person:ada"])])
    events = []
    out = converse_stream("who?", GRAPH, model, on_event=events.append)
    assert "restart" not in [e["event"] for e in events]
    assert out.read == ["person:ada"]
    assert out.content == "Ada and Bea both work on compilers."


def test_a_client_without_deltas_still_streams_whole_pieces():
    """The scripted client ignores on_delta; its thinking and answer arrive as one event
    each."""

    class Muser(ScriptedModel):
        def chat(self, messages, tools=None, **kw):
            self.seen.append(list(messages))
            return Reply(content="Nobody here.", thinking="an empty room")

    events = []
    out = converse_stream("who?", GRAPH, Muser([]), on_event=events.append)
    assert [e["event"] for e in events] \
        == ["thinking", "answer", "restart", "thinking", "answer", "done"]
    assert events[0]["text"] == "an empty room"
    assert events[1]["text"] == "Nobody here."
    assert out.content == "Nobody here."       # it declined to look again, so it stands


def test_show_is_what_the_answer_is_about_not_what_was_opened():
    """The tools' working is not the answer, and lighting the working hides the answer.

    A question about people is answered by reading the topic they share — so `read` holds a
    topic, and the people are named from what it returned. Lighting `read` lit the topic and
    left the people dark, which is what a reader saw on the live page.
    """
    model = ScriptedModel([call("look_at", ids=["topic:compilers"]),
                           call("show", ids=["person:ada", "person:bea"])])
    out = converse("who works on compilers?", GRAPH, model)
    assert out.show == ["person:ada", "person:bea"]
    assert out.read == ["topic:compilers"]
    assert "topic:compilers" not in out.show


def test_show_refuses_an_id_the_model_invented():
    model = ScriptedModel([call("show", ids=["person:ada", "person:nobody", "person:ada"])])
    out = converse("who?", GRAPH, model)
    assert out.show == ["person:ada"]


def test_a_held_entry_the_answer_never_names_is_not_shown():
    """Held entries are told to the model so it knows what the reader is looking at.

    It reads them, so they land in ``read`` — which is exactly how a previous answer's nodes
    kept lighting up a turn later. They reach ``show`` only if the model puts them there.
    """
    model = ScriptedModel([call("look_at", ids=["org:pellard"]),
                           call("show", ids=["person:bea"])])
    out = converse("what about Bea?", GRAPH, model, held=["org:pellard"])
    assert "org:pellard" in out.read
    assert out.show == ["person:bea"]


def test_show_says_what_it_lit_and_is_capped_like_ids():
    model = ScriptedModel([call("show", ids=["person:ada", "person:bea"])])
    out = converse("who?", GRAPH, model, limit=1)
    assert out.show == ["person:ada"]
    assert "lit 2 entries" in out.steps
    one = ScriptedModel([call("show", ids=["person:ada"])])
    assert "lit 1 entry" in converse("who?", GRAPH, one).steps


def test_show_is_offered_as_a_tool_over_any_graph():
    names = [(s.get("function") or {}).get("name") for s, _ in tools_for(GRAPH)]
    assert "show" in names
    # a finder replaces look_up's callable and must not drop the others
    swapped = [(s.get("function") or {}).get("name")
               for s, _ in tools_for(GRAPH, finder=lambda t: [])]
    assert swapped == names


def test_a_streamed_answer_carries_show_too():
    events = []
    model = ScriptedModel([call("show", ids=["person:ada"])])
    out = converse_stream("who?", GRAPH, model, on_event=events.append)
    assert out.show == ["person:ada"]
    assert {e["event"] for e in events} >= {"tool", "tool_result", "done"}


def test_notes_are_never_handed_back_as_the_answer():
    """A model that spends its whole budget planning replies with the plan.

    The empty-reply guard never fired, because the reply was not empty — it was
    "We need to answer: ... Search for X again maybe missing.", printed to the reader
    as the answer. Notes are notes however they arrive.
    """
    from ml_stack.graph.ask import is_working

    notes = ["We need to answer: who can do this. Need people with both.",
             "Let me look up the names first.",
             "From data: one person is interested in it.",
             "Okay, so the question is about two things.",
             "The user asks who could help."]
    answers = ["Ada Lovelace is the one to ask; she has spent years on compilers.",
               "Nobody in the graph does both.",
               "The graph does not answer that.",
               "Two members stand out, and we need not look further than them."]
    assert [is_working(x) for x in notes] == [True] * len(notes)
    assert [is_working(x) for x in answers] == [False] * len(answers)


def test_a_reply_that_is_only_notes_is_asked_again():
    class Planning:
        """Answers with its working, then with an answer when told to."""

        def __init__(self):
            self.asked = 0

        def chat(self, messages, tools=None, **_):
            self.asked += 1
            if tools and self.asked == 1:
                import json
                return Reply(tool_calls=[{"id": "c1", "function": {
                    "name": "look_at", "arguments": json.dumps({"ids": ["person:ada"]})}}])
            if self.asked <= 2:
                return Reply(content="We need to answer: who works on compilers. Need to check.")
            return Reply(content="Ada Lovelace has spent years on compilers.")

    model = Planning()
    out = converse("who works on compilers?", GRAPH, model)
    assert out.content == "Ada Lovelace has spent years on compilers."
    assert "answered with its notes, so was asked again" in out.steps


def test_look_up_takes_several_words_in_one_call():
    """A staffing question needs a lookup per skill; one at a time spends every round."""
    model = ScriptedModel([call("look_up", texts=["compilers", "Bea Marlow"]),
                           call("show", ids=["person:bea"])])
    out = converse("who works on compilers?", GRAPH, model)
    assert set(out.found) == {"topic:compilers", "person:ada", "person:bea"}
    assert "looked up 'compilers', 'Bea Marlow'" in out.steps
    # the single-word form still works, and a word that matches nothing says so
    one = ScriptedModel([call("look_up", text="compilers")])
    assert converse("?", GRAPH, one).found == ["topic:compilers", "person:ada", "person:bea"]


def test_a_finder_is_batched_the_same_way():
    seen = []

    def finder(text):
        seen.append(text)
        return [{"id": "person:ada", "label": "Ada Lovelace", "kind": "person"}]

    tools = tools_for(GRAPH, finder=finder)
    model = ScriptedModel([call("look_up", texts=["one", "two"])])
    out = converse("?", GRAPH, model, tools=tools, finder=finder)
    assert seen == ["one", "two"]
    assert out.found == ["person:ada"]


def test_a_tool_call_written_out_as_prose_is_cut_and_believed():
    """The last turn has no tools, so a model that wanted `show` writes the call instead.

    Measured on the live graph: a good answer arrived with
    `show({"ids":[...]})` welded to the end of it, printed to the reader as part of the
    prose. It is not an answer — but it is the model saying exactly what it meant, and a
    tighter set than asking it again afterwards.
    """
    from ml_stack.graph.ask import spoken_show

    said = ('It would be worthwhile for them to chat about robotics. '
            'show({"ids":["person:ada","person:bea"]})')
    text, ids = spoken_show(said)
    assert text.endswith("chat about robotics.")
    assert ids == ["person:ada", "person:bea"]
    # a harmony wrapper around it comes off too
    wrapped, wrapped_ids = spoken_show('They agree. <|tool_call|>show({"ids": ["person:ada"]})<|end|>')
    assert wrapped == "They agree." and wrapped_ids == ["person:ada"]
    # and an ordinary answer that merely uses the word is left alone
    plain = "She will show you the graph if you ask her nicely."
    assert spoken_show(plain) == (plain, [])


def test_the_written_out_call_is_what_gets_lit():
    """A model that says what to light in words is not asked to say it again."""
    class Writes:
        def __init__(self):
            self.asked = 0

        def chat(self, messages, tools=None, **_):
            self.asked += 1
            if tools and self.asked == 1:
                return Reply(tool_calls=[{"id": "c1", "function": {
                    "name": "look_at", "arguments": json.dumps({"ids": ["topic:compilers"]})}}])
            return Reply(content='Ada and Bea both work on compilers. '
                                 'show({"ids":["person:ada","person:bea"]})')

    import json
    model = Writes()
    out = converse("who works on compilers?", GRAPH, model)
    assert out.show == ["person:ada", "person:bea"]
    assert out.content == "Ada and Bea both work on compilers."
    assert "said what to light in words, so it was not asked again" in out.steps


def test_a_turn_that_stops_searching_can_still_act():
    """Fails when the last call is made with no tools at all.

    Measured against a real model: "I moved to Denver, please fix my entry" went round in
    circles looking the place up, the loop stopped it — and the one call left had nothing to
    reach for, so a member's request to change their own entry came back as "the model did
    not finish an answer". Stopping the searching must not stop the acting.
    """
    import json as _json

    raised = []

    def record(args):
        raised.append(str(args.get("text") or ""))
        return {"recorded": True}

    change = ({"type": "function", "function": {
        "name": "request_change",
        "description": "Ask for something in the graph to be different.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"]}}}, record)

    class GoesInCircles:
        """Searches until it is stopped, then asks for the change it came to ask for."""

        def chat(self, messages, tools=None, **_):
            offered = {str((t.get("function") or {}).get("name")) for t in (tools or [])}
            if "look_up" in offered:
                return Reply(tool_calls=[{"id": "c", "function": {
                    "name": "look_up", "arguments": _json.dumps({"text": "Denver"})}}])
            assert "request_change" in offered, f"nothing left to act with: {offered}"
            return Reply(content="Recorded.", tool_calls=[{"id": "c", "function": {
                "name": "request_change",
                "arguments": _json.dumps({"text": "I moved to Denver"})}}])

    out = converse("I moved to Denver, please fix my entry.", GRAPH, GoesInCircles(),
                   tools=[*tools_for(GRAPH), change], rounds=2)
    assert raised == ["I moved to Denver"], "the request never reached the tool"
    assert "did not finish an answer" not in out.content


def test_an_answer_built_on_nothing_is_sent_back_to_search():
    """A model that answers without touching the graph answered from memory it does not have.

    Measured over the invented community: six of gemma-4-E4B's nine failures were this shape
    — two model calls, a hundred characters of prose, no search. The `show` nudge could not
    catch it, because that one only fires for a turn that *did* search and forgot to say what
    it found. Nothing was correcting the turn that never looked.
    """
    class Blurter:
        def __init__(self):
            self.turns = 0

        def chat(self, messages, tools=None, **_):
            self.turns += 1
            offered = {str((t.get("function") or {}).get("name")) for t in (tools or [])}
            if self.turns == 1:
                return Reply(content="Probably somebody in engineering.")
            if "look_up" in offered:
                return Reply(tool_calls=[{"id": "c1", "function": {
                    "name": "look_up", "arguments": '{"text": "compilers"}'}}])
            return Reply(content="Ada works on compilers.")

    model = Blurter()
    out = converse("who works on compilers?", GRAPH, model)
    assert "topic:compilers" in out.found                    # it went and looked
    assert out.content == "Ada works on compilers."           # and the second answer stands
    assert "answered without looking, so it was sent to look" in out.steps


def test_a_model_that_will_not_search_keeps_the_answer_it_gave():
    """One chance to look, not an argument. Its answer stands rather than being thrown away."""
    class Stubborn:
        def chat(self, messages, tools=None, **_):
            return Reply(content="Nobody here does underwater welding.")

    out = converse("who welds underwater?", GRAPH, Stubborn())
    assert out.content == "Nobody here does underwater welding."
    assert out.found == out.read == out.path == []
    assert "answered without looking, so it was sent to look" not in out.steps


def test_the_tool_descriptions_show_a_call_and_never_use_the_bench_community():
    """Small models need an example, and an example must not be the bench's own answers.

    The descriptions carry worked calls because a model that is only told what a tool is has
    to infer that it should be called. If those examples used the invented community's own
    people, a rising bench score would mean the examples had been memorised rather than the
    convention learned, so the two sets of names are kept apart on purpose.
    """
    from ml_stack.graph.ask import TOOLS
    from ml_stack.graph.community import graph as invented

    nodes = invented()["nodes"]
    theirs = {n["id"].casefold() for n in nodes}
    theirs |= {w for n in nodes for w in str(n.get("label") or "").casefold().split() if w}
    # "repair" leaked in on the first attempt through the look_up example, and it is a topic
    # label here — so labels are split into words rather than matched whole. Ordinary English
    # from the questions ("people", "about") is deliberately not in this set: every example
    # has to be written in some words, and only the graph's own vocabulary can be memorised.

    for schema in TOOLS:
        fn = schema["function"]
        said = fn["description"].casefold()
        assert "{" in said and "example" in said, f"{fn['name']} shows no example call"
        assert fn["name"] in said or "call it" in said
        leaked = sorted(w for w in theirs if w and w in said)
        assert not leaked, f"{fn['name']}'s example uses {leaked}, which the bench asks about"
