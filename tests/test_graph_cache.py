"""An answer is given again only when nothing it was made from has changed.

Every fixture is invented; nothing here reads a real graph or talks to a model.
"""

from __future__ import annotations

import time

import pytest

from ml_stack.graph.ask import Answer
from ml_stack.graph.cache import (PREFIX, asked, digest, fingerprint, forget, kept, recall,
                                  remember)
from ml_stack.graph.store import GraphStore

GRAPH = {
    "nodes": [{"id": "person:iris", "label": "Iris Bellweather", "kind": "person",
               "attrs": {"member": True}, "messages": ["m0"]},
              {"id": "topic:surveying", "label": "surveying", "kind": "topic",
               "attrs": {}, "messages": []}],
    "edges": [{"source": "person:iris", "rel": "experienced_in", "target": "topic:surveying"}],
    "messages": {"m0": {"text": "I survey land.", "ts": "1", "sender": "Iris Bellweather"}},
    "stats": {"messages": 1},
}


@pytest.fixture
def store(tmp_path):
    with GraphStore(tmp_path / "graph.ladybug") as held:
        yield held


def test_a_graph_digest_follows_what_an_answer_could_be_drawn_from():
    same = {**GRAPH, "stats": {"messages": 99}, "meta": {"built": "later"}}
    assert digest(same) == digest(GRAPH)          # a changed count changes nobody's facts

    for change in (
        {**GRAPH, "nodes": [{**GRAPH["nodes"][0], "label": "Iris B."}, GRAPH["nodes"][1]]},
        {**GRAPH, "edges": []},
        {**GRAPH, "messages": {"m0": {"text": "I survey buildings.", "ts": "1"}}},
    ):
        assert digest(change) != digest(GRAPH)

    # node order is not a change: two builds of one graph must agree
    assert digest({**GRAPH, "nodes": list(reversed(GRAPH["nodes"]))}) == digest(GRAPH)


def test_the_fingerprint_covers_everything_that_shapes_an_answer():
    tools = [{"function": {"name": "look_up", "description": "find things",
                           "parameters": {}}}]
    base = dict(graph=GRAPH, model="a-model", system="be brief", tools=tools,
                opening=["person:iris"], limit=25)
    key = fingerprint("who surveys?", **base)
    assert key.startswith(PREFIX)

    # the same question asked the same way, however it was spaced or capitalised
    assert fingerprint("  Who  Surveys? ", **base) == key

    assert fingerprint("who builds?", **base) != key
    assert fingerprint("who surveys?", **{**base, "model": "other"}) != key
    assert fingerprint("who surveys?", **{**base, "system": "be long"}) != key
    assert fingerprint("who surveys?", **{**base, "opening": []}) != key
    assert fingerprint("who surveys?", **{**base, "limit": 60}) != key
    assert fingerprint("who surveys?", **{**base, "graph": {**GRAPH, "edges": []}}) != key

    # the words a tool is described in change what the model does with it, so they must miss
    reworded = [{"function": {"name": "look_up", "description": "find things. Call this "
                                                                "first.", "parameters": {}}}]
    assert fingerprint("who surveys?", **{**base, "tools": reworded}) != key

    # how the question is asked: one flag either way is two answers
    on = fingerprint("who surveys?", **base, ways={"constrain_ids": True})
    off = fingerprint("who surveys?", **base, ways={"constrain_ids": False})
    assert on != off and on != key
    assert fingerprint("who surveys?", **base, ways={"tight": True, "constrain_ids": True}) \
        == fingerprint("who surveys?", **base, ways={"constrain_ids": True, "tight": True})
    assert fingerprint("who surveys?", **base, ways={}) == key, "nothing said is the same key"


def test_a_precomputed_digest_and_a_graph_agree():
    on = digest(GRAPH)
    assert fingerprint("who?", on=on) == fingerprint("who?", graph=GRAPH)


def test_an_answer_is_kept_and_comes_back_whole(store):
    out = Answer(content="Iris surveys land.", show=["person:iris"], read=["person:iris"],
                 steps=["looked up 'survey'"])
    key = fingerprint("who surveys?", graph=GRAPH)
    remember(store, key, out, question="who surveys?")

    back = recall(store, key, kind=Answer)
    assert isinstance(back, Answer)
    assert back.content == "Iris surveys land."
    assert back.show == ["person:iris"] and back.steps == ["looked up 'survey'"]

    assert recall(store, fingerprint("who builds?", graph=GRAPH), kind=Answer) is None


def test_a_kept_answer_stays_out_of_the_graph_the_store_reads_back(store):
    """The cache lives in the same store as the graph and must not become part of it."""
    store.write(GRAPH)
    remember(store, fingerprint("who surveys?", graph=GRAPH), Answer(content="Iris does."))

    read = store.read()
    assert not [k for k in read if k.startswith(PREFIX)]
    assert not [k for k in store.docs() if k.startswith(PREFIX)]
    assert len(kept(store)) == 1                  # but it is there when asked for directly


def test_asked_calls_the_model_once_and_then_never_again(store):
    turns = []

    def fresh():
        turns.append(1)
        return Answer(content="Iris surveys land.", show=["person:iris"])

    making = dict(graph=GRAPH, model="a-model")
    first, was = asked(store, "who surveys?", fresh, kind=Answer, **making)
    assert was is False and len(turns) == 1 and first.content == "Iris surveys land."

    again, was = asked(store, "who surveys?", fresh, kind=Answer, **making)
    assert was is True and len(turns) == 1        # the model was not troubled a second time
    assert again.content == first.content and again.show == first.show

    # change the graph and the same question has to be asked again
    changed = {**GRAPH, "edges": []}
    third, was = asked(store, "who surveys?", fresh, kind=Answer, graph=changed, model="a-model")
    assert was is False and len(turns) == 2


def test_a_remembered_answer_is_replayed_to_a_reader_who_is_streaming(store):
    making = dict(graph=GRAPH, model="a-model")
    asked(store, "who surveys?", lambda: Answer(content="Iris surveys land."), kind=Answer,
          **making)

    events = []
    out, was = asked(store, "who surveys?", lambda: pytest.fail("should not ask"),
                     kind=Answer, on_event=events.append, **making)
    assert was is True
    assert [e["event"] for e in events] == ["answer", "done"]
    assert events[0]["text"] == "Iris surveys land."
    assert events[0]["remembered"] is True and events[1]["remembered"] is True


def test_an_empty_answer_is_not_kept(store):
    """A model that said nothing has not answered, and serving that back forever is worse
    than asking again."""
    out, was = asked(store, "who?", lambda: Answer(content="  "), kind=Answer, graph=GRAPH)
    assert was is False
    assert kept(store) == {}


def test_forgetting(store):
    for q in ("who surveys?", "who builds?", "who sells?"):
        remember(store, fingerprint(q, graph=GRAPH), Answer(content="somebody"), question=q)
    assert len(kept(store)) == 3
    assert forget(store) == 3
    assert kept(store) == {}
    assert forget(store) == 0


def test_an_answer_can_be_made_to_go_stale_by_age(store):
    key = fingerprint("who surveys?", graph=GRAPH)
    remember(store, key, Answer(content="Iris does."))
    assert recall(store, key, kind=Answer) is not None       # no age limit: it stands
    assert recall(store, key, kind=Answer, older_than=3600) is not None
    time.sleep(1.05)
    assert recall(store, key, kind=Answer, older_than=0.5) is None


def test_a_store_that_will_not_answer_is_a_miss_not_a_crash():
    class Broken:
        def get_doc(self, key, default=None):
            raise RuntimeError("no store here")

        def put_doc(self, key, value):
            raise RuntimeError("no store here")

    out, was = asked(Broken(), "who?", lambda: Answer(content="Iris does."), kind=Answer,
                     graph=GRAPH)
    assert was is False and out.content == "Iris does."      # the caller never sees the fault


def test_no_store_at_all_just_asks():
    out, was = asked(None, "who?", lambda: Answer(content="Iris does."), kind=Answer,
                     graph=GRAPH)
    assert was is False and out.content == "Iris does."


def test_what_came_before_is_part_of_the_question():
    """"And where is she based?" means something different after a different question."""
    earlier = [{"role": "user", "content": "who surveys land?"},
               {"role": "assistant", "content": "Iris Bellweather does."}]
    other = [{"role": "user", "content": "who runs the office?"},
             {"role": "assistant", "content": "Otto Vance does."}]

    here = fingerprint("and where is she based?", graph=GRAPH, context=earlier)
    assert here != fingerprint("and where is she based?", graph=GRAPH, context=other)
    assert here != fingerprint("and where is she based?", graph=GRAPH)
    assert here == fingerprint("and where is she based?", graph=GRAPH, context=earlier)


def test_an_answer_the_caller_refuses_is_not_kept(store):
    """Some turns do something as well as say something, and doing it twice is not the same
    as saying it twice."""
    filed = []

    def fresh():
        filed.append("a request")               # the turn had an effect on the world
        return Answer(content="Filed your request.")

    out, was = asked(store, "please change my title", fresh, kind=Answer,
                     keep=lambda _out: not filed, graph=GRAPH)
    assert was is False and kept(store) == {}

    again, was = asked(store, "please change my title", fresh, kind=Answer,
                       keep=lambda _out: not filed, graph=GRAPH)
    assert was is False and len(filed) == 2     # asked again, so it is filed again

    # and an answer the caller does allow is kept as usual
    asked(store, "who surveys?", lambda: Answer(content="Iris does."), kind=Answer,
          keep=lambda _out: True, graph=GRAPH)
    assert len(kept(store)) == 1


def test_a_rebuilt_graph_sweeps_up_the_answers_it_replaced(store):
    """Keys are hashes, so only the digest written beside an answer says which graph it came
    from — and without that the cache grows for the life of the machine."""
    later = {**GRAPH, "edges": []}

    asked(store, "who surveys?", lambda: Answer(content="Iris does."), kind=Answer, graph=GRAPH)
    asked(store, "who else?", lambda: Answer(content="Nobody."), kind=Answer, graph=GRAPH)
    asked(store, "who surveys?", lambda: Answer(content="Nobody now."), kind=Answer,
          graph=later)
    assert len(kept(store)) == 3

    assert forget(store, keeping=digest(later)) == 2          # the two from the old graph
    left = kept(store)
    assert len(left) == 1
    assert next(iter(left.values()))["answer"]["content"] == "Nobody now."

    # the survivor is still served, so sweeping did not cost the answers that are still true
    out, was = asked(store, "who surveys?", lambda: pytest.fail("should not ask"), kind=Answer,
                     graph=later)
    assert was is True and out.content == "Nobody now."


def test_an_entry_with_no_digest_recorded_is_swept(store):
    """Written before digests were kept: there is no way to know it is still true."""
    remember(store, fingerprint("who surveys?", graph=GRAPH), Answer(content="Iris does."))
    assert forget(store, keeping=digest(GRAPH)) == 1
    assert kept(store) == {}
