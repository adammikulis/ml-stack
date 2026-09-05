"""`ml_stack.graph.requests`: a request becomes a proposal, and nothing edits the graph."""

from __future__ import annotations

import json

from ml_stack.graph.ask import draft
from ml_stack.graph.requests import (CHAT_KIND, as_prompt, edge_id, from_chat, key, propose,
                                     read_requests)

GRAPH = {
    "nodes": [{"id": "person:iris", "label": "Iris Bellweather", "kind": "person",
               "attrs": {"member": True}},
              {"id": "topic:looms", "label": "looms", "kind": "topic", "attrs": {}}],
    "edges": [{"source": "person:iris", "rel": "interested_in", "target": "topic:looms"}],
    "messages": {},
}


class Extracting:
    """A model that answers `plan_edits` with the edits given, and holds no conversation."""

    def __init__(self, edits):
        self.edits = edits
        self.calls = 0

    def extract(self, *a, **k):
        self.calls += 1
        return {"edits": self.edits}


class Chatting(Extracting):
    """The same, and it proposes one tool call when offered the tools."""

    def chat(self, messages, tools=None, **k):
        call = {"id": "c1", "function": {"name": "merge_nodes", "arguments": json.dumps(
            {"target": "person:iris", "other": "person:nobody", "reason": "same person"})}}
        return type("Reply", (), {"content": "", "tool_calls": [call]})()


def test_a_request_without_words_is_not_a_request(tmp_path):
    p = tmp_path / "requests.jsonl"
    p.write_text('{"at":"1","text":"drop my location"}\n{"at":"2","text":"  "}\nnot json\n')
    assert [r["at"] for r in read_requests(p)] == ["1"]
    assert read_requests(tmp_path / "missing.jsonl") == []


def test_the_prompt_carries_the_kind_the_claimant_and_what_they_pointed_at():
    out = as_prompt({"kind": "Remove something", "text": "not my topic",
                     "claimedLabel": "Iris Bellweather", "targets": [{"label": "looms"}]})
    assert out.startswith("[Remove something] not my topic")
    assert "Asked by Iris Bellweather" in out and "They pointed at: looms." in out


def test_an_edge_has_a_stable_id():
    assert edge_id(GRAPH["edges"][0]) == "person:iris|interested_in|topic:looms"


def test_a_request_already_proposed_for_is_not_asked_again():
    model = Extracting([])
    request = {"at": "1", "text": "drop looms"}
    done = propose([request], GRAPH, model, done={key(request): {"status": "proposed"}})
    assert model.calls == 0
    assert list(done) == [key(request)]


def test_an_edit_naming_something_not_in_the_graph_is_dropped_and_the_rest_is_kept():
    model = Extracting([
        {"op": "remove", "target": "topic:looms", "name": "", "value": "", "reason": "asked"},
        {"op": "remove", "target": "topic:nonsense", "name": "", "value": "", "reason": "no"}])
    out = propose([{"at": "1", "text": "drop looms", "claimed": "person:iris",
                    "attested": True}], GRAPH, model)
    [made] = out.values()
    assert [e["target"] for e in made["edits"]] == ["topic:looms"]
    assert made["status"] == "proposed" and made["attested"] is True


def test_a_model_that_can_chat_is_asked_the_other_way_round_too():
    out = propose([{"at": "1", "text": "merge my entries", "claimed": "person:iris",
                    "attested": True}], GRAPH, Chatting([]))
    [made] = out.values()
    tool_made = [e for e in made["edits"] if e.get("proposed")]
    assert len(tool_made) == 1 and tool_made[0]["op"] == "merge_nodes"
    assert tool_made[0]["problems"], "an entry that is not there is a problem, listed"


def test_a_chat_request_is_not_attested_and_is_shaped_like_a_form_request():
    k, made = from_chat("drop my topic", GRAPH, Extracting([]), at="2026-01-02T00:00:00Z")
    assert k == "2026-01-02T00:00:00Z|drop my topic"
    assert made["kind"] == CHAT_KIND and made["attested"] is False
    assert made["concerns"], "not attested is a concern"
    assert set(made) == {"at", "kind", "claimed", "claimedLabel", "attested", "concerns",
                         "text", "targets", "edits", "status"}


class Writing:
    def __init__(self, text):
        self.text = text
        self.said = []

    def chat(self, messages, **k):
        self.said.append(messages)
        return type("Reply", (), {"content": self.text})()


def test_draft_writes_a_note_from_what_the_graph_holds_and_needs_two_entries():
    two = {**GRAPH, "nodes": GRAPH["nodes"] + [{"id": "person:tobias", "label": "Tobias Renquist",
                                                "kind": "person", "attrs": {}}]}
    model = Writing("Iris, meet Tobias. You both weave.")
    out = draft(["person:iris", "person:tobias", "person:nobody"], "who weaves?",
                "Iris and Tobias.", two, model, system="Write a note.")
    assert out == {"text": "Iris, meet Tobias. You both weave.",
                   "ids": ["person:iris", "person:tobias"]}
    [messages] = model.said
    assert messages[0] == {"role": "system", "content": "Write a note."}
    assert "Iris Bellweather" in messages[1]["content"] and "who weaves?" in messages[1]["content"]

    assert draft(["person:iris"], "q", "a", two, model)["why"] \
        == "an introduction needs at least two entries"
    assert draft(["person:iris", "person:tobias"], "q", "a", two, Writing("We need to use "
                 "tool search."))["why"] == "the model did not finish a note"
