"""Asking a model a question about a graph.

The model is a stand-in that records what it was given and replays a script of tool calls, so
the tools themselves — which are the part with judgement in them — run for real against a real
graph. What is asserted is what the tools returned and what came back as touched.
"""

from dataclasses import replace

from ml_stack.client import Reply
import json

from ml_stack.graph.ask import (LISTED, Answer, converse, converse_stream, list_kind, look_at,
                                look_up, path_between, tools_for)
from ml_stack.testing import ScriptedModel

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
                return Reply(content="", tool_calls=[{"id": "c1", "function": {
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
                return Reply(content="", tool_calls=[{"id": "c1", "function": {
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
                return Reply(content="", tool_calls=[{"id": "c1", "function": {
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
                return Reply(content="", tool_calls=[{"id": "c1", "function": {
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
    assert "selected 2 entries" in out.steps
    one = ScriptedModel([call("show", ids=["person:ada"])])
    assert "selected 1 entry" in converse("who?", GRAPH, one).steps


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
                return Reply(content="", tool_calls=[{"id": "c1", "function": {
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
                return Reply(content="", tool_calls=[{"id": "c1", "function": {
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
                return Reply(content="", tool_calls=[{"id": "c", "function": {
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
                return Reply(content="", tool_calls=[{"id": "c1", "function": {
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
    # Words, not fragments: "Brayfield Survey Co" would otherwise flag every use of "co",
    # which matches inside "costs". A name short enough to collide by accident is not a
    # name anyone could memorise the answers from.
    #
    # And names, not sentences. An opportunity is labelled with a phrase -- "surveying a
    # site before it is bought" -- whose ordinary words ("before", "site") are nobody's
    # answer to anything. Only labels short enough to *be* a name contribute their words.
    theirs |= {w for n in nodes
               if len(str(n.get("label") or "").split()) <= 3
               for w in str(n.get("label") or "").casefold().split() if len(w) > 3}
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


def test_the_tools_can_be_said_briefly_or_at_length():
    """What a model needs to be told depends on the model. The worked examples took
    gemma-4-E4B from 17% to 70% recall and cost gpt-oss-120b twenty points over the same
    questions, so both exist and the caller chooses -- there is no answering that from
    first principles."""
    from ml_stack.graph.ask import TERSE, TOOLS, tools_for

    assert [t["function"]["name"] for t in TERSE] \
        == [t["function"]["name"] for t in TOOLS], "the same six tools, said differently"
    def shape(schema):
        """The callable shape: names, types and what is required -- not the prose."""
        params = schema["function"]["parameters"]
        return (sorted(params.get("required") or []),
                {name: prop.get("type") for name, prop in params["properties"].items()})

    for terse, full in zip(TERSE, TOOLS, strict=True):
        assert shape(terse) == shape(full), \
            "only the words differ; a model that reads either must call either identically"
        assert len(terse["function"]["description"]) < len(full["function"]["description"])

    graph = {"nodes": [], "edges": [], "messages": {}}
    assert [s["function"]["name"] for s, _fn in tools_for(graph, terse=True)] \
        == [s["function"]["name"] for s, _fn in tools_for(graph)]
    # tight is the asking now and hands out copies; loose -- tight=False, the control --
    # is the sets themselves. Either way look_up is word for word its own: tight changes
    # what show says and nothing else.
    assert tools_for(graph, terse=True, tight=False)[0][0] is TERSE[0]
    assert tools_for(graph, tight=False)[0][0] is TOOLS[0]
    assert tools_for(graph, terse=True)[0][0] == TERSE[0]
    assert tools_for(graph)[0][0] == TOOLS[0]


def test_a_turn_stops_as_soon_as_it_has_said_what_to_light():
    """`show` is the last thing a turn does. A round after it is a round trip spent to be
    told the same thing."""
    class Model:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, **_):
            self.calls += 1
            offered = {str((t.get("function") or {}).get("name")) for t in (tools or [])}
            if self.calls == 1 and "look_at" in offered:
                return Reply(content="", tool_calls=[{"id": "a", "function": {
                    "name": "look_at", "arguments": '{"ids": ["person:ada"]}'}}])
            if self.calls == 2 and "show" in offered:
                return Reply(content="", tool_calls=[{"id": "b", "function": {
                    "name": "show", "arguments": '{"ids": ["person:ada"]}'}}])
            return Reply(content="Ada works on compilers.")

    model = Model()
    out = converse("who?", GRAPH, model)
    assert out.show == ["person:ada"]
    assert "said what to light, so the searching stopped" in out.steps
    assert model.calls == 3, f"one search, one show, one answer -- not {model.calls}"


def test_a_turn_that_shows_and_keeps_looking_is_not_finished():
    """Showing in the same breath as another search is not a finished turn."""
    class Model:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, **_):
            self.calls += 1
            offered = {str((t.get("function") or {}).get("name")) for t in (tools or [])}
            if self.calls == 1 and "look_up" in offered:
                return Reply(content="", tool_calls=[
                    {"id": "a", "function": {"name": "show",
                                             "arguments": '{"ids": ["person:ada"]}'}},
                    {"id": "b", "function": {"name": "look_up",
                                             "arguments": '{"text": "compilers"}'}}])
            return Reply(content="Ada works on compilers.")

    model = Model()
    out = converse("who?", GRAPH, model)
    assert "said what to light, so the searching stopped" not in out.steps
    assert len(out.found) > 0, "the search in that round still happened"


def test_going_in_circles_costs_a_handful_of_calls_and_stops():
    """The counter is checked at the top of the round, which is before the next request goes
    out -- so hitting the limit costs no extra round trip. Checked by measurement: moving
    the check to just after dispatch changed nothing, five calls either way, and the change
    was removed rather than kept as an unjustified difference."""
    class Stubborn:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, **_):
            self.calls += 1
            offered = {str((t.get("function") or {}).get("name")) for t in (tools or [])}
            if "look_up" in offered:
                return Reply(content="", tool_calls=[{"id": "x", "function": {
                    "name": "look_up", "arguments": '{"text": "compilers"}'}}])
            return Reply(content="Nothing more to find.")

    model = Stubborn()
    out = converse("who?", GRAPH, model, rounds=10)
    assert "stopped searching in circles" in out.steps
    assert model.calls == 5, f"a circle costs five calls here, not {model.calls}"


def test_list_kind_lists_a_kind_most_mentioned_first_and_names_the_kinds_when_it_misses():
    """The question no search reaches: nothing is labelled "company", so a small model
    searching for one finds nothing and gives up. Listing a kind is a capability, not a
    wording, and a wrong guess at the kind's name is answered with the right ones."""
    out = list_kind(GRAPH, "person")
    assert out["kind"] == "person" and out["total"] == 2
    assert [e["id"] for e in out["entries"]] == ["person:ada", "person:bea"]
    assert out["entries"][0] == {"id": "person:ada", "label": "Ada Lovelace", "mentions": 4}
    # case and number are not the model's problem: Orgs is org, topics is topic
    assert [e["id"] for e in list_kind(GRAPH, "Orgs")["entries"]] == ["org:pellard"]
    assert list_kind(GRAPH, "Topics")["kind"] == "topic"
    # a miss says so and names every kind there is, biggest first, so the next call is right
    missed = list_kind(GRAPH, "company")
    assert missed == {"none": "no kind 'company'", "kinds": {"person": 2, "org": 1, "topic": 1}}
    assert list_kind(GRAPH, "")["kinds"] == missed["kinds"]


def test_list_kind_reads_the_kind_off_the_id_when_a_node_does_not_say_and_is_capped():
    graph = {"nodes": [{"id": "place:turin", "label": "Turin", "mentions": 1},
                       {"id": "place:harrowgate", "label": "Harrowgate", "mentions": 1},
                       {"id": "place:selby", "label": "Selby", "mentions": 5},
                       {"id": "loose", "label": "no kind at all"}],
             "edges": [], "messages": {}}
    out = list_kind(graph, "places", limit=2)
    # most mentioned first, then by label; the total says what the cap left out
    assert [e["id"] for e in out["entries"]] == ["place:selby", "place:harrowgate"]
    assert out["total"] == 3
    assert list_kind(graph, "nothing")["kinds"] == {"place": 3}
    assert LISTED >= 40, "a community's every organisation fits in one listing"


def test_a_model_that_lists_a_kind_gets_the_entries_read_back():
    model = ScriptedModel([call("list_kind", kind="org"),
                           call("show", ids=["org:pellard"])])
    out = converse("which companies do people here work for?", GRAPH, model)
    assert out.found == ["org:pellard"]
    assert out.show == ["org:pellard"]
    assert "listed 1 of kind 'org'" in out.steps
    tool_turns = [m for turn in model.seen for m in turn if m.get("role") == "tool"]
    assert any("org:pellard" in m["content"] and "Pellard Foundry" in m["content"]
               for m in tool_turns)
    # a wrong guess is a step too, and it counts as having gone looking
    wrong = ScriptedModel([call("list_kind", kind="company")])
    out = converse("which companies?", GRAPH, wrong)
    assert "found no kind 'company'" in out.steps
    assert "answered without looking, so it was sent to look" not in out.steps


def test_listing_a_kind_is_a_search_and_is_taken_away_on_the_final_turn():
    """The final turn is for answering; a tool that goes looking is not offered on it."""
    offered_by_call: list[set[str]] = []

    class Lister:
        def chat(self, messages, tools=None, **_):
            offered = {str((t.get("function") or {}).get("name")) for t in (tools or [])}
            offered_by_call.append(offered)
            if "list_kind" in offered:
                return Reply(content="", tool_calls=[{"id": "x", "function": {
                    "name": "list_kind", "arguments": '{"kind": "org"}'}}])
            return Reply(content="Pellard Foundry is the only company here.")

    out = converse("which companies?", GRAPH, Lister(), rounds=2)
    assert out.content == "Pellard Foundry is the only company here."
    assert "list_kind" in offered_by_call[0]
    assert "list_kind" not in offered_by_call[2], "the answering turn still offered a search"
    assert "show" in offered_by_call[2], "the tool that acts was taken away with the searches"


def test_show_is_found_by_name_after_the_tools_were_reordered():
    """Fails when the closing nudge indexes TOOLS by position: adding list_kind before show
    would offer the model list_kind and ask it to show, and nothing would light up."""
    from ml_stack.graph.ask import TERSE, TOOLS, _schema

    assert TOOLS[3]["function"]["name"] != "show", "the reorder this test exists for"
    assert _schema("show")["function"]["name"] == "show"
    assert _schema("show", TERSE)["function"]["name"] == "show"
    offered_last: list[set[str]] = []

    class Reader:
        """Reads once, answers in words, and shows only when show is all it is offered --
        which is the closing nudge, and nowhere else."""

        def chat(self, messages, tools=None, **_):
            offered = {str((t.get("function") or {}).get("name")) for t in (tools or [])}
            offered_last.append(offered)
            if len(offered_last) == 1:
                return Reply(content="", tool_calls=[{"id": "a", "function": {
                    "name": "look_at", "arguments": '{"ids": ["person:ada"]}'}}])
            if offered == {"show"}:
                return Reply(content="", tool_calls=[{"id": "b", "function": {
                    "name": "show", "arguments": '{"ids": ["person:ada"]}'}}])
            return Reply(content="Ada works on compilers.")

    out = converse("who?", GRAPH, Reader())
    assert len(offered_last) == 3, "one read, one answer, one nudge to say what to light"
    assert offered_last[-1] == {"show"}
    assert out.show == ["person:ada"]


def test_the_shortlist_comes_before_the_question_as_candidates_to_verify():
    """Measured on gemma-4-E4B: eight likely entries as the *last* message after the
    question, phrased "use them if they answer it", took it from 58% F1 to 33% -- it echoed
    the list instead of selecting from it. So the list arrives before the question, as
    something to check, and the question is the last thing the model reads."""
    events = []
    model = ScriptedModel([call("show", ids=["person:ada"])])
    out = converse_stream("who works on compilers?", GRAPH, model, on_event=events.append,
                          opening=["person:ada", "person:ghost"])
    first = model.seen[0]
    assert [m["role"] for m in first] == ["system", "user", "user"]
    assert first[-1]["content"] == "who works on compilers?"
    handed = first[1]["content"]
    assert handed.startswith("A search turned up these entries; some may be irrelevant.")
    assert "before trusting them" in handed and "ignore the rest" in handed
    assert "Ada Lovelace (person)" in handed
    assert "Use them if they answer it" not in handed
    assert out.found == ["person:ada"] and out.steps[0] == "was handed 1 to start from"
    assert events[0] == {"event": "tool", "name": "shortlist", "detail": "1 to start from"}


def test_an_empty_answer_says_why_from_the_last_reply():
    """Measured: an empty answer is almost never the token budget. The same failing question
    at n_predict 2048 and 6144 both came back finish_reason=stop with 628-767 characters of
    reasoning and no answer, and nothing printed that, so the ceiling was raised again and
    proved nothing. The steps now say what the last reply actually did."""

    class OnlyThinks(ScriptedModel):
        def chat(self, messages, tools=None, **kw):
            self.seen.append(list(messages))
            if tools and self.script:
                import json
                name, args = self.script.pop(0)
                return Reply(content="", tool_calls=[{"id": "c1", "function": {
                    "name": name, "arguments": json.dumps(args)}}])
            return Reply(content="", thinking=SCRATCH, finish_reason="stop")

    out = converse("who?", GRAPH, OnlyThinks([call("look_at", ids=["person:ada"])]))
    assert "did not finish an answer" in out.content
    assert out.steps[-1] == f"no answer: finish_reason=stop, thinking {len(SCRATCH)} chars, " \
                            "answer 0 chars"
    # a truncated one says so too, which is the one case where the ceiling is the fix
    class CutOff(OnlyThinks):
        def chat(self, messages, tools=None, **kw):
            out = super().chat(messages, tools=tools, **kw)
            return Reply(content="", thinking="x" * 50, finish_reason="length") \
                if out.tool_calls is None else out

    out = converse("who?", GRAPH, CutOff([call("look_at", ids=["person:ada"])]))
    assert out.steps[-1] == "no answer: finish_reason=length, thinking 50 chars, answer 0 chars"


def test_planning_welded_to_the_answer_without_a_space_is_cut_off():
    """Measured: one sentence of planning arrives stuck to the front of a good answer with
    no space after the full stop. Splitting on ". " alone keeps them as one sentence, and
    then cutting the note cuts the answer with it."""
    from ml_stack.graph.ask import without_notes

    answer = ("Grace Hopper and Ada Lovelace are the two members who have spent years on "
              "compilers.")
    assert without_notes("We need to use tool search." + answer) == answer
    assert without_notes("We need to use tool search. " + answer) == answer
    # and an answer with no note on it is left exactly as it was
    assert without_notes(answer) == answer


def _tiny_png() -> bytes:
    """A one-pixel PNG, built by hand so the test owns its own fixture."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)          # 1x1, 8-bit RGB
    pixel = zlib.compress(b"\x00\x00\x00\x00")                    # filter byte + one pixel
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixel) + chunk(b"IEND", b"")


def _picture_tool(images):
    def look(args):
        return {"url": str(args.get("url") or ""), "title": "Tinsley Works", "text": "a kiln",
                "_images": images}

    return ({"type": "function", "function": {
        "name": "web_look",
        "description": "Fetch a page and what it looks like.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}}, look)


def test_pictures_a_tool_brings_back_are_shown_in_a_message_of_their_own():
    """A tool result cannot carry an image through llama.cpp, so the pictures come out of
    the result before it is encoded and follow it as a user message the vision model can
    see. The encoder must never meet the bytes."""
    import json
    import pytest

    pytest.importorskip("PIL")
    model = ScriptedModel([call("web_look", url="https://example.invalid/kiln")])
    events = []
    out = converse_stream("what does the page show?", GRAPH, model, on_event=events.append,
                          tools=[*tools_for(GRAPH), _picture_tool([_tiny_png()])])
    assert out.steps == ["used web_look"]
    turn = model.seen[-1]
    tool_at = next(i for i, m in enumerate(turn) if m.get("role") == "tool")
    assert "_images" not in turn[tool_at]["content"]
    assert json.loads(turn[tool_at]["content"])["title"] == "Tinsley Works"
    shown = turn[tool_at + 1]
    assert shown["role"] == "user"
    parts = shown["content"]
    assert parts[0] == {"type": "text",
                        "text": "What web_look returned for the call above, as seen:"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert {"event": "tool_result", "name": "web_look", "count": 1} in events


def test_a_picture_that_cannot_be_prepared_is_said_not_sent():
    """A model told to look at nothing answers about nothing, confidently."""
    model = ScriptedModel([call("web_look", url="https://example.invalid/kiln")])
    out = converse("what does the page show?", GRAPH, model,
                   tools=[*tools_for(GRAPH), _picture_tool([b"not a picture at all"])])
    assert out.steps[0] == "used web_look"
    assert out.steps[1].startswith("web_look returned 1 image and none could be shown: ")
    turn = model.seen[-1]
    tool_at = next(i for i, m in enumerate(turn) if m.get("role") == "tool")
    assert "_images" not in turn[tool_at]["content"]
    assert not any(isinstance(m.get("content"), list) for m in turn), "nothing was shown"
    # an empty list of pictures is not a picture, and is simply dropped
    plain = ScriptedModel([call("web_look", url="https://example.invalid/kiln")])
    converse("?", GRAPH, plain, tools=[*tools_for(GRAPH), _picture_tool([])])
    assert not any(isinstance(m.get("content"), list) for m in plain.seen[-1])


def test_bytes_anywhere_else_in_a_tool_result_never_reach_the_encoder():
    def leaks(args):
        return {"raw": b"\x00\x01\x02", "note": "fine"}

    schema = {"type": "function", "function": {
        "name": "leaky", "description": "returns bytes",
        "parameters": {"type": "object", "properties": {}, "required": []}}}
    model = ScriptedModel([call("leaky")])
    out = converse("?", GRAPH, model, tools=[*tools_for(GRAPH), (schema, leaks)])
    assert out.steps == ["used leaky"]
    tool_turn = next(m for m in model.seen[-1] if m.get("role") == "tool")
    assert '"<3 bytes>"' in tool_turn["content"]


def test_the_web_tools_are_searches_and_go_away_with_the_others():
    """`web_search`, `web_read` and `web_look` look for something, so the last turn -- the one
    that exists to end the searching -- must not offer them either. Measured before this: the
    quiet turn offered {'request_change', 'show', 'web_read', 'web_search'}.
    Mutation: drop the web names from SEARCHING."""
    from ml_stack.graph.ask import SEARCHING

    assert {"web_search", "web_read", "web_look"} <= SEARCHING
    assert "show" not in SEARCHING


def _staffing_graph(people=10):
    """A topic with more people joined to it than a rich hit carries, plus one person joined
    to nothing and one place, so the cap, the order and the kinds can each be checked."""
    nodes = [{"id": "topic:welding", "kind": "topic", "label": "welding", "mentions": 20,
              "attrs": {}, "messages": []},
             {"id": "place:ambleford", "kind": "place", "label": "Ambleford", "mentions": 5,
              "attrs": {}, "messages": []},
             {"id": "org:tinsley", "kind": "org", "label": "Tinsley Works", "mentions": 2,
              "attrs": {}, "messages": []},
             {"id": "person:loner", "kind": "person", "label": "Orla Quist", "mentions": 9,
              "attrs": {}, "messages": []}]
    edges = []
    for n in range(people):
        pid = f"person:w{n}"
        nodes.append({"id": pid, "kind": "person", "label": f"Welder {n:02d}",
                      "mentions": n, "attrs": {}, "messages": []})
        # both directions, so an edge into the topic counts the same as one out of it
        if n % 2:
            edges.append({"source": pid, "target": "topic:welding", "rel": "interested_in"})
        else:
            edges.append({"source": "topic:welding", "target": pid, "rel": "has_member"})
    edges.append({"source": "topic:welding", "target": "org:tinsley", "rel": "related"})
    edges.append({"source": "person:w3", "target": "place:ambleford", "rel": "lives_in"})
    return {"nodes": nodes, "edges": edges, "messages": {}}


def test_with_rich_off_look_up_returns_exactly_what_it_always_did():
    """The comparability guarantee: the answer cache fingerprints the tool descriptions, so
    with rich off look_up is invisible -- the same schema object under the loose asking
    (tight=False, the control), word for word the same under the tight default, which
    touches show alone, and the same keys in the JSON the model reads."""
    import json

    from ml_stack.graph.ask import RICH_SENTENCE, TOOLS

    assert tools_for(GRAPH, tight=False)[0][0] is TOOLS[0]
    assert tools_for(GRAPH)[0][0] == TOOLS[0]
    assert RICH_SENTENCE not in TOOLS[0]["function"]["description"]
    model = ScriptedModel([call("look_up", text="compilers")])
    converse("?", GRAPH, model)
    handed = [m for m in model.seen[-1] if m.get("role") == "tool"]
    rows = json.loads(handed[0]["content"])
    assert rows and all(set(r) == {"id", "label", "kind"} for r in rows)
    assert all(set(r) == {"id", "label", "kind"} for r in look_up(GRAPH, "compilers"))


def test_a_rich_topic_hit_says_why_and_brings_its_people_most_mentioned_first_and_capped():
    from ml_stack.graph.ask import JOINED_HITS, joined_people

    graph = _staffing_graph()
    find = tools_for(graph, rich=True)[0][1]
    rows = find({"text": "welding"})
    assert rows[0]["id"] == "topic:welding"
    assert rows[0]["score"] == 4 and rows[0]["matched"] == ["label"]
    joined = rows[0]["joined"]
    assert len(joined) == JOINED_HITS == 8
    # ten people are joined; the eight most mentioned come, best first, from both directions
    assert [j["id"] for j in joined] == [f"person:w{n}" for n in range(9, 1, -1)]
    assert joined[0] == {"id": "person:w9", "label": "Welder 09"}
    assert "org:tinsley" not in {j["id"] for j in joined}, "only people are brought"
    # a place brings its one person; an org with nobody says so, plainly
    assert joined_people(graph, "place:ambleford") == [{"id": "person:w3", "label": "Welder 03"}]
    assert joined_people(graph, "org:tinsley") == []
    assert find({"text": "tinsley"})[0]["joined"] == []


def test_a_rich_person_hit_has_no_joined_list():
    graph = _staffing_graph()
    find = tools_for(graph, rich=True)[0][1]
    rows = find({"text": "Orla"})
    assert rows[0]["id"] == "person:loner"
    assert "joined" not in rows[0]
    assert set(rows[0]) == {"id", "label", "kind", "score", "matched"}


def test_a_finder_that_does_not_say_why_still_brings_the_people():
    """The app's finder returns plain rows; the wrapper adds what it can read from the graph
    and leaves out what it would only be guessing at."""
    graph = _staffing_graph()

    def finder(text):
        return [{"id": "topic:welding", "label": "welding", "kind": "topic"},
                {"id": "person:w1", "label": "Welder 01", "kind": "person"}]

    plain = tools_for(graph, finder=finder)[0][1]({"text": "x"})
    assert all(set(r) == {"id", "label", "kind"} for r in plain)
    rich = tools_for(graph, finder=finder, rich=True)[0][1]({"text": "x"})
    assert set(rich[0]) == {"id", "label", "kind", "joined"}
    assert len(rich[0]["joined"]) == 8
    assert set(rich[1]) == {"id", "label", "kind"}
    # a finder that does say why keeps its word, and the finder's own rows are untouched
    said = [{"id": "topic:welding", "label": "welding", "kind": "topic",
             "score": 0.033, "matched": ["label", "words"]}]
    out = tools_for(graph, finder=lambda t: said, rich=True)[0][1]({"text": "x"})
    assert out[0]["matched"] == ["label", "words"] and out[0]["score"] == 0.033
    assert "joined" in out[0] and "joined" not in said[0]


def test_the_rich_sentence_is_only_in_the_rich_schema():
    from ml_stack.graph.ask import RICH_SENTENCE, TERSE, TOOLS

    for terse, base in ((False, TOOLS), (True, TERSE)):
        rich = tools_for(GRAPH, terse=terse, rich=True)[0][0]
        assert rich is not base[0]
        assert rich["function"]["description"].endswith(RICH_SENTENCE)
        assert RICH_SENTENCE not in base[0]["function"]["description"]
        assert rich["function"]["parameters"] == base[0]["function"]["parameters"]
        assert rich["function"]["description"] == \
            base[0]["function"]["description"] + " " + RICH_SENTENCE
        # the other four are the same words
        for r, b in zip(tools_for(GRAPH, terse=terse, rich=True)[1:],
                        tools_for(GRAPH, terse=terse)[1:], strict=True):
            assert r[0] == b[0]


def test_converse_rich_reaches_the_tool():
    import json

    model = ScriptedModel([call("look_up", text="compilers")])
    out = converse("who does compilers?", GRAPH, model, rich=True)
    assert out.found[0] == "topic:compilers"
    handed = [m for m in model.seen[-1] if m.get("role") == "tool"]
    rows = json.loads(handed[0]["content"])
    topic = next(r for r in rows if r["id"] == "topic:compilers")
    assert topic["matched"] == ["label"] and topic["score"] == 4
    assert topic["joined"] == [{"id": "person:ada", "label": "Ada Lovelace"},
                               {"id": "person:bea", "label": "Bea Marlow"}]
    # the streamed path takes the same flag
    events = []
    streamed = ScriptedModel([call("look_up", text="compilers")])
    converse_stream("who does compilers?", GRAPH, streamed, on_event=events.append, rich=True)
    handed = [m for m in streamed.seen[-1] if m.get("role") == "tool"]
    assert "joined" in json.loads(handed[0]["content"])[0]


# -- tight: light only what answers the question ----------------------------------------------

MANY = {
    "nodes": [
        {"id": f"person:p{i}", "kind": "person", "label": name, "mentions": 1, "attrs": {}}
        for i, name in enumerate(("Wren Tallis", "Hollis Marne", "Ida Pellow", "Bram Oakes",
                                  "Nell Farrow", "Tobin Ashe", "Marek Voss", "Cass Lindley"),
                                 start=1)
    ],
    "edges": [],
    "messages": {},
}
EIGHT = [n["id"] for n in MANY["nodes"]]


class SayingModel(ScriptedModel):
    """A scripted model whose answer is chosen, and that keeps what tools it was offered."""

    def __init__(self, script, answer):
        super().__init__(script)
        self.answer = answer
        self.offered: list[list[dict]] = []

    def chat(self, messages, tools=None, **kw):
        self.offered.append(list(tools or []))
        reply = super().chat(messages, tools=tools, **kw)
        if not reply.tool_calls:
            reply = replace(reply, content=self.answer)
        return reply


def test_tight_is_the_default_asking_and_tight_off_is_the_old_one():
    """Tight is how everything asks now. `tight=False` is the control: the asking the
    ranking runs and the answer cache fingerprinted, so it is still the same schemas (not
    copies), the same nudge and the same system prompt, byte for byte."""
    from ml_stack.graph.ask import (SHOW_PARAGRAPH, SYSTEM, TERSE, TIGHT_NUDGE,
                                    TIGHT_SENTENCE, TIGHT_SHOW, TIGHT_SHOW_PARAGRAPH,
                                    TIGHT_SYSTEM_SENTENCE, TOOLS, tools_for)

    for got, base in zip(tools_for(GRAPH, tight=False), TOOLS, strict=True):
        assert got[0] is base
    for got, base in zip(tools_for(GRAPH, terse=True, tight=False), TERSE, strict=True):
        assert got[0] is base
    assert TIGHT_SENTENCE not in TOOLS[-1]["function"]["description"]
    assert TIGHT_SENTENCE not in TERSE[-1]["function"]["description"]

    loose = SayingModel([call("look_at", ids=["person:ada"])], "Ada Lovelace does compilers.")
    out = converse("who?", GRAPH, loose, tight=False)
    assert out.read == ["person:ada"] and not out.show
    assert loose.seen[0][0]["content"] == SYSTEM
    assert loose.seen[-1][-1]["content"] == (
        "Now call show once with the ids of the entries your answer is about — everyone and "
        "everything you named in it, including any you named from a quote. Nothing you "
        "opened and did not write about.")
    assert loose.offered[-1][0]["function"]["description"] \
        == TOOLS[-1]["function"]["description"]

    # and the default, asked nothing at all, is the tight asking: the tight system prompt,
    # the tight nudge, the tight show
    model = SayingModel([call("look_at", ids=["person:ada"])], "Ada Lovelace does compilers.")
    converse("who?", GRAPH, model)
    assert model.seen[0][0]["content"] == (
        SYSTEM.replace(SHOW_PARAGRAPH, TIGHT_SHOW_PARAGRAPH) + " " + TIGHT_SYSTEM_SENTENCE)
    assert model.seen[-1][-1]["content"] == TIGHT_NUDGE
    assert model.offered[-1][0]["function"]["description"] == TIGHT_SHOW


def test_tight_changes_what_show_says_on_a_copy_of_every_set():
    from ml_stack.graph.ask import (RICH_SENTENCE, TERSE, TIGHT_SENTENCE, TIGHT_SHOW,
                                    TIGHT_SHOW_TERSE, TOOLS, tools_for)

    for terse, base, want in ((False, TOOLS, TIGHT_SHOW), (True, TERSE, TIGHT_SHOW_TERSE)):
        got = tools_for(GRAPH, terse=terse, tight=True)
        show = got[-1][0]
        assert show is not base[-1]
        assert show["function"]["name"] == "show"
        assert show["function"]["description"] == want
        assert TIGHT_SENTENCE in want
        assert show["function"]["parameters"] == base[-1]["function"]["parameters"]
        # the other four are the same words
        for g, b in zip(got[:-1], base[:-1], strict=True):
            assert g[0] == b
    # rich and tight compose: look_up says what its hits carry, show says what to light
    both = tools_for(GRAPH, rich=True, tight=True)
    assert both[0][0]["function"]["description"].endswith(RICH_SENTENCE)
    assert both[-1][0]["function"]["description"] == TIGHT_SHOW
    # the full description still shows a call, like every other, and names the tool
    assert "{" in TIGHT_SHOW and "example" in TIGHT_SHOW.casefold() and "show" in TIGHT_SHOW


def test_tight_nudge_and_system_carry_the_new_sentences_only_when_asked():
    from ml_stack.graph.ask import (SYSTEM, TIGHT_NUDGE, TIGHT_SHOW, TIGHT_SHOW_TERSE,
                                    TIGHT_SYSTEM_SENTENCE, tools_for)
    from ml_stack.graph.ask import SHOW_PARAGRAPH, TIGHT_SHOW_PARAGRAPH

    model = SayingModel([call("look_at", ids=["person:ada"])], "Ada Lovelace does compilers.")
    converse("who?", GRAPH, model, tight=True)
    assert model.seen[0][0]["content"] == SYSTEM.replace(SHOW_PARAGRAPH, TIGHT_SHOW_PARAGRAPH) + " " + TIGHT_SYSTEM_SENTENCE
    assert TIGHT_NUDGE.startswith("Now call show once with only the entries that answer the")
    assert "which-of-a-kind" in TIGHT_NUDGE and "chain" in TIGHT_NUDGE
    assert model.seen[-1][-1]["content"] == TIGHT_NUDGE
    assert model.offered[-1] == [next(s for s, _ in tools_for(GRAPH, tight=True)
                                      if s["function"]["name"] == "show")]
    assert model.offered[-1][0]["function"]["description"] == TIGHT_SHOW
    # a caller's own tools are told too: the terse set handed in -- built loose, as a
    # caller assembling its own set has it -- gets the terse tight show
    handed = SayingModel([call("look_at", ids=["person:ada"])], "Ada Lovelace does compilers.")
    converse("who?", GRAPH, handed, tools=tools_for(GRAPH, terse=True, tight=False),
             tight=True)
    first = handed.offered[0]
    assert next(s for s in first if s["function"]["name"] == "show")["function"]["description"] \
        == TIGHT_SHOW_TERSE
    assert TIGHT_SYSTEM_SENTENCE not in SYSTEM


def test_tight_caps_show_at_six_keeping_what_the_prose_names_most_named_first():
    from ml_stack.graph.ask import LIT_TIGHT

    assert LIT_TIGHT == 6
    said = "Marek Voss leads it, with Ida Pellow beside him; ask Marek Voss first."
    model = SayingModel([call("look_at", ids=EIGHT), call("show", ids=EIGHT)], said)
    out = converse("who leads?", MANY, model, tight=True)
    assert out.show == ["person:p7", "person:p3", "person:p1", "person:p2", "person:p4",
                        "person:p5"]
    assert "cut 2 of 8 lit" in out.steps
    assert "selected 8 entries" in out.steps, "what the model asked for is still on record"
    # the same script, asked loose (tight=False, the control): all eight, capped only by LIT
    plain = SayingModel([call("look_at", ids=EIGHT), call("show", ids=EIGHT)], said)
    out = converse("who leads?", MANY, plain, tight=False)
    assert out.show == EIGHT
    assert not [s for s in out.steps if s.startswith("cut ")]


def test_tight_keeps_what_a_tool_returned_and_drops_what_none_did():
    """Known is what any tool returned: Bea was *found* by look_up though never read, and
    the first tight rule dropped her -- measured 2026-09-02, that rule cut a listed place.
    A name no tool ever returned is the guess (the `made` case) and goes.
    Mutation: build `seen` from `out.read` alone, or skip the drop."""
    from ml_stack.graph.ask import converse, look_up

    import copy

    graph = copy.deepcopy(GRAPH)
    # somebody in the graph whom no tool returns and nothing read is joined to
    never = "person:cass"
    graph["nodes"].append({"id": never, "kind": "person", "label": "Cass Lindley",
                           "mentions": 1, "attrs": {}})
    found = {r["id"] for r in look_up(graph, "compilers")}
    assert "person:bea" in found and never not in found
    said = "Ada Lovelace and Bea Marlow both do compilers, and so does Cass Lindley."
    script = [call("look_up", text="compilers"), call("look_at", ids=["person:ada"]),
              call("show", ids=["person:ada", "person:bea", never])]
    out = converse("who does compilers?", graph, SayingModel(list(script), said), tight=True)
    assert "person:bea" in out.found and "person:bea" not in out.read
    assert out.show == ["person:ada", "person:bea"], "found counts as known; the guess goes"
    assert "dropped 1 unread from show" in out.steps
    # asked loose (tight=False, the control): everything the model asked for
    out = converse("who does compilers?", graph, SayingModel(list(script), said), tight=False)
    assert out.show == ["person:ada", "person:bea", never]
    assert not [s for s in out.steps if s.startswith("dropped ")]


def test_the_streamed_path_takes_tight_too():
    events = []
    said = "Ada Lovelace works on compilers, and so does Bea Marlow."
    model = SayingModel([call("look_up", text="compilers"), call("look_at", ids=["person:ada"]),
                         call("show", ids=["person:ada", "person:bea"])], said)
    out = converse_stream("who?", GRAPH, model, on_event=events.append, tight=True)
    assert out.show == ["person:ada", "person:bea"], "found by look_up, so known"
    assert not [s for s in out.steps if s.startswith("dropped ")]
    assert events[-1] == {"event": "done"}


def test_the_tight_system_prompt_no_longer_says_every_name_belongs_in_show():
    """The base prompt says every name written belongs in show; tight's copy says only the
    entries that answer belong, and the base is unchanged. Mutation: drop the replace."""
    from ml_stack.graph.ask import SHOW_PARAGRAPH, SYSTEM, TIGHT_SHOW_PARAGRAPH

    assert SHOW_PARAGRAPH in SYSTEM
    tight = SYSTEM.replace(SHOW_PARAGRAPH, TIGHT_SHOW_PARAGRAPH)
    assert "Every one you named belongs" not in tight and "usually one to three" in tight


def test_tight_spares_a_listing_from_the_cap_and_keeps_what_a_tool_returned(tmp_path):
    """Measured 2026-09-02 on Flash-Next: the cap cut a thirteen-company listing to six
    alphabetically and threw away three expected ones, and the unread rule dropped a place
    the model had listed and seen joined to the people it read. Mutation: cut without
    sparing `out.listed`, or build `seen` from `out.read` alone."""
    from ml_stack.graph.ask import LIT_TIGHT, _joined_to

    graph = {"nodes": [{"id": f"org:o{i}", "kind": "org", "label": f"Org {i}"} for i in range(9)]
             + [{"id": "person:p", "kind": "person", "label": "Wren Halloway"},
                {"id": "place:c", "kind": "place", "label": "Calderwick"}],
             "edges": [{"source": "person:p", "target": "place:c", "rel": "based_in"}]}
    assert _joined_to(graph, ["person:p"]) == {"place:c"}
    assert LIT_TIGHT == 6
    # the cap logic, exercised directly on an Answer-shaped object
    from ml_stack.graph.ask import Answer
    out = Answer(content="Org 0 through Org 8 all employ people here.")
    out.listed = [f"org:o{i}" for i in range(9)]
    out.show = list(out.listed)
    listed = set(out.listed)
    rest = [i for i in out.show if i not in listed]
    assert rest == [] and len(out.show) == 9, "a listing is never cut"


# ------------------------------------------------------------ of any length

from types import SimpleNamespace  # noqa: E402

from ml_stack.graph.ask import (EARLIER, RECALLED, SHOW_PARAGRAPH,  # noqa: E402
                                SYSTEM, TIGHT_SHOW_PARAGRAPH, TIGHT_SYSTEM_SENTENCE)

# The system prompt as the default asking sends it: tight. `SYSTEM` itself is what the
# control -- tight=False -- still sends, byte for byte.
TIGHT_SYSTEM = (SYSTEM.replace(SHOW_PARAGRAPH, TIGHT_SHOW_PARAGRAPH) + " "
                + TIGHT_SYSTEM_SENTENCE)

WINDOW_TURNS = [{"role": "user", "content": "who is Ada Lovelace?"},
                {"role": "assistant", "content": "Ada Lovelace is an analyst at Pellard Foundry."},
                {"role": "user", "content": "and Bea?"},
                {"role": "assistant", "content": "Bea Marlow works on compilers too."}]


def test_without_a_summary_or_recall_the_messages_are_byte_for_byte_what_they_were():
    """The ranking runs and the answer cache depend on this: with the new kwargs absent,
    the messages are the system prompt, the window in order, the shortlist, the question --
    the prompt being the tight one, which is what the default asking sends."""
    before = ScriptedModel([])
    converse("who else?", GRAPH, before, turns=WINDOW_TURNS)
    assert before.seen[0] == [{"role": "system", "content": TIGHT_SYSTEM}, *WINDOW_TURNS,
                              {"role": "user", "content": "who else?"}]
    # and the control still sends SYSTEM itself
    loose = ScriptedModel([])
    converse("who else?", GRAPH, loose, turns=WINDOW_TURNS, tight=False)
    assert loose.seen[0][0] == {"role": "system", "content": SYSTEM}

    explicit = ScriptedModel([])
    converse("who else?", GRAPH, explicit, turns=WINDOW_TURNS, summary=None, recalled=())
    assert explicit.seen == before.seen

    empty = ScriptedModel([])
    converse("who else?", GRAPH, empty, turns=WINDOW_TURNS, summary="   ",
             recalled=[{"role": "user", "content": ""}, None])
    assert empty.seen == before.seen

    listed = ScriptedModel([])
    converse("who else?", GRAPH, listed, turns=WINDOW_TURNS, opening=["person:ada"])
    seen = listed.seen[0]
    assert seen[:1 + len(WINDOW_TURNS)] == before.seen[0][:1 + len(WINDOW_TURNS)]
    assert seen[-2]["content"].startswith("A search turned up these entries")
    assert seen[-1] == {"role": "user", "content": "who else?"}


def test_the_summary_goes_first_then_the_recalled_then_the_window_then_the_shortlist():
    """Order, and the shapes each may arrive in: a string, a mapping, a Turn-like object."""
    model = ScriptedModel([])
    recalled = [{"role": "user", "content": "did Ada ever move?"},
                SimpleNamespace(role="assistant", text="Ada moved to Turin in the spring."),
                {"role": "user", "text": "a mapping with text rather than content"}]
    converse("where is she now?", GRAPH, model, turns=WINDOW_TURNS,
             summary=SimpleNamespace(role="summary", text="Ada is the subject.\n  Rests on person:ada."),
             recalled=recalled, opening=["person:ada"])
    seen = model.seen[0]
    assert seen[0] == {"role": "system", "content": TIGHT_SYSTEM}
    assert seen[1] == {"role": "user", "content": EARLIER + "Ada is the subject. Rests on person:ada."}
    assert seen[2:5] == [
        {"role": "user", "content": RECALLED + "did Ada ever move?"},
        {"role": "assistant", "content": RECALLED + "Ada moved to Turin in the spring."},
        {"role": "user", "content": RECALLED + "a mapping with text rather than content"}]
    assert seen[5:9] == WINDOW_TURNS
    assert seen[9]["content"].startswith("A search turned up these entries")
    assert seen[10] == {"role": "user", "content": "where is she now?"} and len(seen) == 11

    # a bare string is a summary too, and the streaming path assembles the same messages
    streamed = ScriptedModel([])
    converse_stream("where is she now?", GRAPH, streamed, on_event=lambda e: None,
                    turns=WINDOW_TURNS, summary="Ada is the subject.", recalled=recalled[:1])
    assert streamed.seen[0][1] == {"role": "user", "content": EARLIER + "Ada is the subject."}
    assert streamed.seen[0][2]["content"] == RECALLED + "did Ada ever move?"
    assert streamed.seen[0][3:7] == WINDOW_TURNS


def test_a_follow_up_resolves_from_the_window_alone():
    """"And where is she based?" has its "she" in the window and nowhere else: no summary,
    nothing recalled, and the model reads Ada off the turn before and opens her entry."""
    model = ScriptedModel([call("look_at", ids=["person:ada"])], answer="Ada is based in Turin.")
    out = converse("and where is she based?", GRAPH, model, turns=WINDOW_TURNS[:2])
    assert model.seen[0][1:3] == WINDOW_TURNS[:2]
    assert model.seen[0][3] == {"role": "user", "content": "and where is she based?"}
    assert out.read == ["person:ada"] and "Turin" in model.told()
    assert out.content == "Ada is based in Turin."


def test_an_answer_says_which_model_answered_and_what_every_call_spent():
    """`Answer.spent` is noted from every reply of the loop: the model the server named, the
    calls, the tokens read, kept and written, and the draft head's acceptance -- so a person
    testing an answer sees the cost without running the bench."""
    from ml_stack.client.chat import Reply

    raw = {"model": "tiny-Q4.gguf", "usage": {"prompt_tokens": 500, "completion_tokens": 30},
           "timings": {"prompt_n": 200, "cache_n": 300, "prompt_ms": 50.0, "predicted_ms": 150.0,
                       "draft_n": 10, "draft_n_accepted": 7}}

    class Model:
        def __init__(self):
            self.turn = 0

        def chat(self, messages, tools=None, **kw):
            self.turn += 1
            if self.turn == 1:
                return Reply(content=None, raw=raw, finish_reason="tool_calls", tool_calls=[
                    {"id": "c1", "type": "function",
                     "function": {"name": "look_up", "arguments": json.dumps({"text": "Ada"})}}])
            return Reply(content="Ada works on compilers.", raw=raw, finish_reason="stop")

    out = converse("what does Ada do?", GRAPH, Model())
    assert out.model == "tiny-Q4.gguf"
    assert out.spent.calls >= 2 and out.spent.tool_calls >= 1
    assert out.spent.read_tokens == 200 * out.spent.calls
    assert out.spent.cached_tokens == 300 * out.spent.calls
    assert out.spent.acceptance == 0.7 and out.spent.drafted
    assert out.spent.finish == "stop" and not out.spent.truncated
    assert out.spent.seconds >= 0
    # what the slot held is the server's count, per call, and the peak is what bounds users
    assert out.spent.context_peak == 200 + 300 and out.spent.context_last == 500
    # what filled the prompt, by part -- the window and summary are absent here, so the
    # parts are the system text, the tool schemas, the question, what the tools returned,
    # and the answer
    assert {"system", "tools", "question", "tool_results", "answer"} <= set(out.spent.parts)
    assert all(n > 0 for n in out.spent.parts.values())
    assert "window" not in out.spent.parts and "summary" not in out.spent.parts


def test_the_parts_of_a_prompt_are_counted_when_a_window_and_a_summary_are_sent():
    from ml_stack.client.chat import Reply

    class Model:
        def chat(self, messages, tools=None, **kw):
            return Reply(content="Ada.", raw={}, finish_reason="stop")

    turns = [{"role": "user", "content": "who is Ada?"}, {"role": "assistant", "content": "A person."}]
    out = converse("and Bea?", GRAPH, Model(), turns=turns, summary="Earlier they asked about Ada.")
    assert out.spent.parts.get("window", 0) > 0 and out.spent.parts.get("summary", 0) > 0
    # no timings from this fake: the peak falls back to the usage, which is absent, so 0
    assert out.spent.context_peak == 0

# -- look_around and reach: one fat call for a model that reads faster than it writes -------
#
# Measured 2026-09-02 on Qwen3.8-Flash-Next: 89-95% recall, 5-9 tool calls a question at
# about 2k new tokens each, a result read back at ~390 tok/s and an answer written at ~35.
# Half the wall clock was reading, so the cheaper question is fewer, fatter calls -- which
# is what these two are. E4B and E2B have the opposite profile and are left on the default,
# which is why `reach` is None unless a caller asks.

AROUND_GRAPH = {
    "nodes": [
        {"id": "topic:ceramics", "kind": "topic", "label": "ceramics", "mentions": 9,
         "attrs": {}, "messages": ["m1"]},
        {"id": "person:wren", "kind": "person", "label": "Wren Halloway", "mentions": 5,
         "attrs": {"role": "potter", "location": "Ambleford"}, "messages": ["m2"]},
        {"id": "person:hollis", "kind": "person", "label": "Hollis Fen", "mentions": 2,
         "attrs": {}, "messages": ["m3"]},
        {"id": "org:tinsley", "kind": "org", "label": "Tinsley Kilnworks", "mentions": 1,
         "attrs": {"type": "company"}, "messages": []},
    ],
    "edges": [
        {"source": "person:wren", "target": "topic:ceramics", "rel": "fires", "weight": 2},
        {"source": "person:hollis", "target": "topic:ceramics", "rel": "teaches", "weight": 9},
        {"source": "person:wren", "target": "org:tinsley", "rel": "works_at", "weight": 1},
    ],
    "messages": {"m1": {"text": "the studio kiln cracked again"},
                 "m2": {"text": "I fire the kiln most mornings."},
                 "m3": {"text": "I teach the Tuesday class."}},
}


def test_look_around_reads_a_whole_neighbourhood_in_one_call():
    """The five calls it replaces: a look_up finds the topic, and everyone on it comes back
    with the relation, their kind, their id and a line of their own words -- so a staffing
    question never spends a call per neighbour."""
    from ml_stack.graph.ask import look_around

    text = look_around(AROUND_GRAPH, ["topic:ceramics"])
    assert text.startswith("- ceramics [topic:ceramics] (topic)")
    assert 'said: "the studio kiln cracked again"' in text
    # the heaviest edge first, then by id -- the same graph reads out the same way twice
    assert text.index("Hollis Fen") < text.index("Wren Halloway")
    assert "<- teaches Hollis Fen [person:hollis] (person)" in text
    assert "<- fires Wren Halloway [person:wren] (person)" in text
    # a neighbour's own words, so the answer can quote what it never looked up
    assert 'said: "I teach the Tuesday class."' in text
    # the direction is kept: "Wren works_at Tinsley" is not "Tinsley works_at Wren"
    out = look_around(AROUND_GRAPH, ["person:wren"])
    assert "-> works_at Tinsley Kilnworks [org:tinsley] (company)" in out
    assert "location: Ambleford" in out and "role: potter" in out
    assert look_around(AROUND_GRAPH, ["person:nobody"]) == ""


def test_look_around_goes_a_second_hop_when_asked_and_never_loops():
    from ml_stack.graph.ask import look_around

    one = look_around(AROUND_GRAPH, ["topic:ceramics"])
    two = look_around(AROUND_GRAPH, ["topic:ceramics"], hops=2)
    assert one.count("\n- ") == 0, "one hop is one entry read out, with its joins under it"
    # the neighbours become entries in their own right, each once however many edges lead back
    assert two.count("\n- ") == 2 and two.startswith("- ceramics [topic:ceramics]")
    assert "- Wren Halloway [person:wren] (person)" in two
    assert "-> works_at Tinsley Kilnworks" in two, "a second hop reaches what one did not"


def test_a_reach_packs_whole_entries_with_their_quotes_and_the_default_packs_nothing():
    """A budget is a reading budget. Four entries entire answer a question that twelve with
    their quotes cut out do not, so what a budget drops is entries, never words -- and the
    most-mentioned go first. Without one nothing is packed at all: what a caller asked for
    is what it gets, byte for byte as it always was."""
    from ml_stack.graph.ask import look_at

    ids = ["person:hollis", "person:wren", "topic:ceramics"]
    whole = look_at(AROUND_GRAPH, ids)
    assert look_at(AROUND_GRAPH, ids, budget=None) == whole
    assert look_at(AROUND_GRAPH, ids, budget=10_000) == whole, "it all fits, so nothing moves"
    tight = look_at(AROUND_GRAPH, ids, budget=24)
    assert len(tight) < len(whole)
    assert tight.startswith("- ceramics"), "the most-mentioned entry, whole"
    assert 'said: "the studio kiln cracked again"' in tight, "with its quote, not without"
    assert "- Hollis Fen" not in tight, "the entry that did not fit, not its words"


def test_a_reach_lets_list_kind_read_out_more_than_its_fixed_forty():
    """The cap was a guess at what a result may cost. A model whose context is cheap wants
    every organisation there is, so a budget replaces the cap rather than joining it."""
    from ml_stack.graph.ask import LISTED, list_kind

    many = {"nodes": [{"id": f"org:o{i}", "kind": "org", "label": f"Org {i}", "mentions": i}
                      for i in range(60)],
            "edges": [], "messages": {}}
    assert len(list_kind(many, "org")["entries"]) == LISTED
    assert list_kind(many, "org")["total"] == 60
    wide = list_kind(many, "org", budget=4000)
    assert len(wide["entries"]) == 60, "a budget that fits them all reads them all"
    narrow = list_kind(many, "org", budget=40)
    assert 0 < len(narrow["entries"]) < LISTED and narrow["total"] == 60
    assert narrow["entries"][0]["mentions"] == 59, "the tail is cut, not the order"


def test_a_reach_cuts_a_tool_message_by_tokens_and_none_cuts_by_characters():
    """The flat 6000 characters is what every run so far measured, so it is what a
    conversation with no reach still gets."""
    from ml_stack.client.tokens import estimate_tokens
    from ml_stack.graph.ask import CUT, _cut

    long = "the kiln cracked again. " * 2000
    assert _cut(long, None) == long[:CUT]
    assert _cut("short", None) == "short"
    held = _cut(long, 200)
    assert estimate_tokens(held) <= 200 and len(held) < len(long)
    assert _cut("short", 200) == "short", "under the budget nothing is touched"


def test_converse_offers_look_around_and_a_reach_reaches_its_result():
    """The plumbing: `reach` is a conversation's, so it has to arrive at the tools and at
    the cut on the tool message. Mutation: pass it to `tools_for` and not to `_cut`, and a
    caller's own fat tool result is still trimmed at 6000 characters."""
    model = ScriptedModel([call("look_around", ids=["topic:ceramics"])])
    out = converse("who could help with a cracked kiln?", AROUND_GRAPH, model)
    assert out.read == ["topic:ceramics"], "what it was given, it read"
    # every entry the result named it found -- so the score, the cap and tight count them
    assert set(out.found) == {"topic:ceramics", "person:hollis", "person:wren"}
    assert out.steps == ["looked around 1 entry"]
    handed = [m for turn in model.seen for m in turn if m.get("role") == "tool"]
    assert "person:hollis" in handed[0]["content"]

    # with a reach small enough to matter, the same call comes back shorter
    small = ScriptedModel([call("look_around", ids=["topic:ceramics"])])
    converse("who?", AROUND_GRAPH, small, reach=12)
    short = [m for turn in small.seen for m in turn if m.get("role") == "tool"]
    assert len(short[0]["content"]) < len(handed[0]["content"])
    assert converse_stream("who?", AROUND_GRAPH,
                           ScriptedModel([call("look_around", ids=["topic:ceramics"])]),
                           on_event=lambda e: None, reach=8000).read == ["topic:ceramics"]


def test_look_around_is_a_search_and_goes_away_on_the_last_turn():
    """A model that may still look around will look around instead of answering, which is
    the failure the last turn exists to end."""
    from ml_stack.graph.ask import SEARCHING, TOOLS

    assert "look_around" in SEARCHING
    named = {t["function"]["name"] for t in TOOLS}
    assert "look_around" in named
    model = SayingModel([call("look_around", ids=["topic:ceramics"]),
                         call("look_around", ids=["person:wren"])], "Hollis Fen teaches it.")
    converse("who?", AROUND_GRAPH, model, rounds=1)
    assert not any("look_around" in {t["function"]["name"] for t in offered}
                   for offered in model.offered[1:]), "taken away with the other searches"


def test_a_neighbour_only_ever_seen_inside_a_neighbourhood_may_still_be_lit():
    """Tight drops a name no tool ever returned. What look_around read out *is* returned --
    the whole point of one fat call -- so a model may write about a neighbour and select it
    without a look_at of its own. Mutation: leave look_around out of `note(out.found, ...)`
    and the right answer is dropped as a guess."""
    model = SayingModel([call("look_around", ids=["topic:ceramics"], hops=2),
                         call("show", ids=["person:hollis", "org:tinsley"])],
                        "Hollis Fen teaches the class and Wren Halloway fires for "
                        "Tinsley Kilnworks.")
    out = converse("who could help with a cracked kiln?", AROUND_GRAPH, model)
    # `org:tinsley` is two hops out: nothing was read that it is joined to, so the only
    # thing that makes it known is that look_around read it out
    assert "org:tinsley" in out.found and "org:tinsley" not in out.read
    assert out.show == ["person:hollis", "org:tinsley"]
    assert not any("unread" in step for step in out.steps)


# -- all the lookups in one turn ------------------------------------------------------------

class ParallelModel:
    """A model that asks for several tools in one reply, as llama-server returns them.

    `ScriptedModel` issues one call per turn, which cannot show whether the loop runs every
    call a reply carried or only the first. ``batches`` is a list of turns, each a list of
    ``(name, args)``; a turn is issued when every tool in it is on offer, and once they are
    spent it answers in words. ``chat`` carries `Client.chat`'s signature.
    """

    def __init__(self, batches, answer=ScriptedModel.ANSWER):
        self.batches = [list(one) for one in batches]
        self.answer = answer
        self.seen: list[list[dict]] = []

    def chat(self, messages, *, tools=None, tool_choice="auto", timeout=None,
             on_delta=None, **extra):
        self.seen.append(list(messages))
        offered = {str((t.get("function") or {}).get("name")) for t in (tools or [])}
        if self.batches and all(name in offered for name, _args in self.batches[0]):
            turn = self.batches.pop(0)
            return Reply(content="", tool_calls=[
                {"id": f"c{n}", "function": {"name": name,
                                             "arguments": json.dumps(args)}}
                for n, (name, args) in enumerate(turn)])
        return Reply(content=self.answer)


def test_every_tool_call_in_one_reply_is_run_before_the_next_model_turn():
    """A reply may carry several calls at once, and all of them are the model's question:
    running the first and dropping the rest would lose two thirds of a batched turn and
    look, from the outside, exactly like a model that had not batched. Each is answered
    with a tool message of its own, and the three of them are one round."""
    model = ParallelModel([[call("look_at", ids=["person:ada"]),
                            call("look_at", ids=["person:bea"]),
                            call("look_up", text="Pellard")]])
    out = converse("who works on compilers?", GRAPH, model)
    assert out.read == ["person:ada", "person:bea"]
    assert "org:pellard" in out.found
    answered = [m for turn in model.seen for m in turn if m.get("role") == "tool"]
    # three results, and the ids in them, so nothing was run and then thrown away
    assert len({m["tool_call_id"] for m in answered}) == 3
    assert any("Bea Marlow" in m["content"] for m in answered)
    assert out.rounds == 1, "one reply, one round -- however many calls it carried"


def test_reading_three_entries_in_one_call_is_one_round_not_three():
    """What `batch` is for. The same question answered by naming everything in one look_up
    and one look_around costs three rounds through the model where one entry at a time
    costs seven, and the round is the round trip -- which is where the wall clock goes."""
    model = ScriptedModel([call("look_up", texts=["compilers", "Ada", "Pellard"]),
                           call("look_around", ids=["person:ada", "person:bea",
                                                    "topic:compilers"]),
                           call("show", ids=["person:ada", "person:bea"])])
    out = converse("who works on compilers?", GRAPH, model, batch=True)
    assert out.rounds == 3, "look_up, look_around, show"
    assert out.spent.calls == 4, "three rounds and the turn that wrote the answer"
    assert out.spent.tool_calls == 3
    assert out.show == ["person:ada", "person:bea"]

    slow = ScriptedModel([call("look_up", texts=["compilers"]),
                          call("look_at", ids=["person:ada"]),
                          call("look_at", ids=["person:bea"]),
                          call("look_at", ids=["topic:compilers"]),
                          call("show", ids=["person:ada", "person:bea"])])
    assert converse("who works on compilers?", GRAPH, slow, batch=True).rounds == 5


def test_a_turn_that_read_one_entry_while_more_were_found_is_told_to_read_the_rest():
    from ml_stack.graph.ask import BATCH_NUDGE

    one_at_a_time = ScriptedModel([call("look_up", text="compil"),
                                   call("look_at", ids=["person:ada"]),
                                   call("look_at", ids=["person:bea"])])
    out = converse("who works on compilers?", GRAPH, one_at_a_time, batch=True)
    said = [m for turn in one_at_a_time.seen for m in turn if m.get("role") == "user"]
    assert any(m["content"] == BATCH_NUDGE for m in said)
    assert "asked it to read the rest in one call" in out.steps

    both_at_once = ScriptedModel([call("look_up", text="compil"),
                                  call("look_at", ids=["person:ada", "person:bea"])])
    converse("who works on compilers?", GRAPH, both_at_once, batch=True)
    assert not any(m.get("content") == BATCH_NUDGE
                   for turn in both_at_once.seen for m in turn)

    # and never without asking for it: the nudge is a message in every prompt after it
    off = ScriptedModel([call("look_up", text="compil"),
                         call("look_at", ids=["person:ada"]),
                         call("look_at", ids=["person:bea"])])
    converse("who works on compilers?", GRAPH, off)
    assert not any(m.get("content") == BATCH_NUDGE for turn in off.seen for m in turn)


def test_the_batch_sentence_and_the_worked_calls_are_said_only_when_asked_for():
    """On copies, like rich and tight: with the flag off the descriptions and the system
    prompt are byte for byte what the ranking runs measured."""
    from ml_stack.graph.ask import (BATCH_EXAMPLES, BATCH_SYSTEM_SENTENCE, TOOLS,
                                    tools_for)

    said = ScriptedModel([])
    converse("who?", GRAPH, said, batch=True)
    system = said.seen[0][0]["content"]
    assert BATCH_SYSTEM_SENTENCE in system

    quiet = ScriptedModel([])
    converse("who?", GRAPH, quiet)
    assert BATCH_SYSTEM_SENTENCE not in quiet.seen[0][0]["content"]

    batched = {t["function"]["name"]: t["function"]["description"]
               for t, _fn in tools_for(GRAPH, batch=True)}
    for name, example in BATCH_EXAMPLES.items():
        assert batched[name].endswith(example)
        assert example not in _schema_text(TOOLS, name), "the base set is untouched"
    assert "person:hollis" in batched["look_at"], "a three-entry call, written out"


def _schema_text(schemas, name):
    return next(t["function"]["description"] for t in schemas
                if t["function"]["name"] == name)


# -- the kind the question asked for ---------------------------------------------------------

def test_asked_kinds_reads_the_question_word():
    """The table, over the questions the bench actually asks. A question whose own words
    settle the kind is filtered; one naming several kinds or none is left alone, because a
    filter that guesses wrong empties the selection."""
    from ml_stack.graph.ask import asked_kinds
    from ml_stack.graph.community import QUESTIONS

    table = {
        "Who fixes machines?": {"person"},
        "Who knows about robotics?": {"person"},
        "Who is based in Dunmore?": {"person"},
        "Someone who can sell things": {"person"},
        "Who could introduce Iris Bellweather to a lawyer?": {"person"},
        "Who here is called Vance?": {"person"},
        "I need two people to build a healthcare AI prototype. Who?": {"person"},
        "Which companies do people here work for?": {"org"},
        "Which company employs the lawyer?": {"org"},
        "Which company does surveying?": {"org"},
        "Where does Alan Turing work?": {"org"},
        "Where is Nell Ashgrove based?": {"place"},
        "Which places do the people who do repair live in?": {"place"},
        "Where do the people who went to Makers Night live?": {"place"},
        "What events come up?": {"event"},
        "Which events did Grace Hopper go to?": {"event"},
        "What openings are there?": {"opportunity"},
        "Any work going for a surveyor?": {"opportunity"},
        "How are Otto Vance and Charles Babbage connected?": None,
        "Tell me about Otto Vance.": None,
    }
    assert len(table) == 20
    asked = {q["q"] for q in QUESTIONS}
    for question, wanted in table.items():
        assert question in asked, f"{question!r} is not one of the bench's questions"
        assert asked_kinds(question) == wanted, question

    # the question that mentions a person and asks about a subject: what comes last is
    # what is being asked
    assert asked_kinds("Somebody is selling a machine and needs help. "
                       "What do they need?") is None
    # several kinds named at once, and a topic question said plainly
    assert asked_kinds("who and what is here?") is None
    assert asked_kinds("Which people and companies are here?") is None
    assert asked_kinds("What topics come up here?") == {"topic"}
    assert asked_kinds("") is None


def test_a_who_question_does_not_light_the_topic_it_was_found_through():
    model = ScriptedModel([call("look_up", text="compil"),
                           call("show", ids=["person:ada", "person:bea",
                                             "topic:compilers"])])
    out = converse("who knows about compilers?", GRAPH, model, kinds=True)
    assert out.show == ["person:ada", "person:bea"]
    assert any("another kind" in step for step in out.steps)

    loose = ScriptedModel([call("look_up", text="compil"),
                           call("show", ids=["person:ada", "person:bea",
                                             "topic:compilers"])])
    assert converse("who knows about compilers?", GRAPH, loose).show == \
        ["person:ada", "person:bea", "topic:compilers"], "off unless asked for"


def test_a_question_naming_no_kind_or_several_filters_nothing():
    model = ScriptedModel([call("path_between", from_id="person:ada", to_id="person:bea"),
                           call("show", ids=["person:ada", "topic:compilers",
                                             "person:bea"])])
    out = converse("How are Ada Lovelace and Bea Marlow connected?", GRAPH, model,
                   kinds=True)
    assert out.show == ["person:ada", "topic:compilers", "person:bea"]
    assert not any("another kind" in step for step in out.steps)


def test_a_listing_is_exempt_from_the_kind_filter_and_an_empty_selection_is_never_made():
    """`list_kind` is exempt from the cap for the same reason it is exempt here: a
    which-of-a-kind question has as many right answers as there are entries. And a filter
    that would light nothing at all is not applied -- a blank graph is worse than a loose
    selection."""
    listing = ScriptedModel([call("list_kind", kind="topic"),
                             call("show", ids=["topic:compilers", "org:pellard"])])
    out = converse("Who is here?", GRAPH, listing, kinds=True)
    assert out.show == ["topic:compilers"], "the listed entry stays; the other kind goes"

    everything = ScriptedModel([call("look_up", text="compil"),
                                call("show", ids=["topic:compilers"])])
    kept = converse("who knows about compilers?", GRAPH, everything, kinds=True)
    assert kept.show == ["topic:compilers"]
    assert not any("another kind" in step for step in kept.steps)


# -- the whole graph at a glance ---------------------------------------------------------------

def test_summarise_reads_out_the_counts_the_top_entries_and_the_busiest_joins():
    from ml_stack.graph.ask import summarise

    text = summarise(GRAPH)
    assert text.splitlines()[0] == \
        "This graph holds 4 entries and 3 joins: 2 person, 1 org, 1 topic."
    assert "person (2), most mentioned first:" in text
    # each entry with its id in brackets, as look_around writes them, and a line of its words
    assert "- Ada Lovelace [person:ada], analyst, Turin said: \"I am Ada" in text
    assert "- Pellard Foundry [org:pellard], company" in text
    assert "Busiest joins: interested_in (2), works_at (1)." in text
    assert summarise({"nodes": [], "edges": []}) == "This graph holds 0 entries and 0 joins."


def test_the_summary_tool_is_offered_only_when_asked_for_and_goes_away_last():
    from ml_stack.graph.ask import SEARCHING, TOOLS, tools_for

    assert "summarise" not in {t["function"]["name"] for t in TOOLS}
    assert "summarise" not in {s["function"]["name"] for s, _fn in tools_for(GRAPH)}
    assert "summarise" in {s["function"]["name"]
                           for s, _fn in tools_for(GRAPH, summary=True)}
    # a model that may still summarise will summarise instead of answering
    assert "summarise" in SEARCHING


def test_a_broad_question_calls_summarise_once_and_selects_the_top_entries():
    model = ScriptedModel([call("summarise"),
                           call("show", ids=["person:ada", "topic:compilers"])])
    out = converse("what is this community about?", GRAPH, model, summary_tool=True)
    assert out.steps.count("read the graph at a glance") == 1
    assert out.show == ["person:ada", "topic:compilers"]
    # what the summary read out counts as found, so it may be selected without a look_at
    assert set(out.found) == {"person:ada", "person:bea", "topic:compilers", "org:pellard"}
    assert not out.read
    told = model.told()
    assert "This graph holds 4 entries" in told


def test_a_broad_question_routes_to_the_summary_only_where_it_is_offered():
    """The examples are `graph.route`'s, and they are added only when the tool is: a broad
    question routed to something the model was never given is a question with no tool."""
    from ml_stack.graph.ask import SUMMARY_PROMPTS, TOOL_PROMPTS, prompts_for, routing_prompts
    from ml_stack.graph.route import rank

    def words(text):
        return {w.strip(".,?!") for w in text.casefold().split()}

    def fake_embedder(texts, **kw):
        vocab = sorted({w for t in texts for w in words(t)})
        return [[1.0 if w in words(t) else 0.0 for w in vocab] for t in texts]

    def routed(question, **kw):
        return rank(question, base_url="http://nowhere.invalid", model="pretend",
                    embedder=fake_embedder, prompts=routing_prompts(**kw))

    assert routed("what is this community about?", summary=True).order[0] == "summarise"
    assert routed("what does this group actually do?", summary=True).order[0] == "summarise"
    # a particular name is still a search, summary or not
    assert routed("who fixes machines?", summary=True).order[0] == "look_up"
    assert "summarise" not in routing_prompts()
    assert "summarise" not in TOOL_PROMPTS, \
        "TOOL_PROMPTS is read as 'every tool the model has'; the summary is offered on ask"
    assert prompts_for("summarise") == SUMMARY_PROMPTS
