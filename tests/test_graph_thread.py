"""A conversation kept in the graph, joined to what each turn drew on.

Every fixture is invented; nothing reads a real graph or talks to a model.
"""

from __future__ import annotations

import pytest

from ml_stack.graph.ask import Answer
from ml_stack.graph.store import GraphStore
from ml_stack.graph.thread import (drew_on, follow, forget_thread, of_node, recent,
                                   remember_turn, threads, turn_of)

GRAPH = {
    "nodes": [{"id": "person:iris", "label": "Iris Bellweather", "kind": "person",
               "attrs": {}, "messages": []},
              {"id": "person:otto", "label": "Otto Vance", "kind": "person",
               "attrs": {}, "messages": []},
              {"id": "topic:surveying", "label": "surveying", "kind": "topic",
               "attrs": {}, "messages": []}],
    "edges": [{"source": "person:iris", "rel": "experienced_in", "target": "topic:surveying"}],
    "messages": {},
}


@pytest.fixture
def store(tmp_path):
    with GraphStore(tmp_path / "graph.ladybug") as held:
        held.write(GRAPH)
        yield held


def test_drew_on_keeps_the_ways_apart_rather_than_merging_them():
    """Found, read, travelled and named are four different things about one entry."""
    answer = Answer(content="Iris surveys land.", found=["person:iris", "topic:surveying"],
                    read=["topic:surveying"], path=[], show=["person:iris"])
    assert drew_on(answer) == {"found": ["person:iris", "topic:surveying"],
                               "read": ["topic:surveying"], "shown": ["person:iris"]}
    assert drew_on(Answer(content="Nobody.")) == {}
    # a mapping works too, which is what a payload over HTTP looks like
    assert drew_on({"shown": ["person:otto"], "read": []}) == {"shown": ["person:otto"]}


def test_turns_chain_in_the_order_they_were_said(store):
    remember_turn(store, thread="t1", role="user", text="who surveys land?")
    remember_turn(store, thread="t1", role="assistant", text="Iris does.",
                  drew={"shown": ["person:iris"]})
    remember_turn(store, thread="t1", role="user", text="and who else?")

    said = follow(store, "t1")
    assert [t.seq for t in said] == [1, 2, 3]
    assert [t.role for t in said] == ["user", "assistant", "user"]
    assert said[1].drew == {"shown": ["person:iris"]}
    assert said[0].drew == {} and said[2].drew == {}

    # a second conversation does not mix in
    remember_turn(store, thread="t2", role="user", text="separate")
    assert [t.text for t in follow(store, "t2")] == ["separate"]
    assert len(follow(store, "t1")) == 3


def test_a_turn_cannot_join_an_entry_the_graph_does_not_hold(store):
    """The same rule the tool loop follows: a conversation must not invent entries."""
    turn = remember_turn(store, thread="t1", role="assistant", text="Someone does.",
                         drew={"shown": ["person:iris", "person:nobody"]})
    assert turn.drew == {"shown": ["person:iris"]}
    assert turn_of(store, turn.id).drew == {"shown": ["person:iris"]}


def test_what_have_we_said_about_this_person(store):
    """The question a flat log cannot answer."""
    remember_turn(store, thread="t1", role="assistant", text="Iris surveys land.",
                  drew={"shown": ["person:iris"], "read": ["topic:surveying"]})
    remember_turn(store, thread="t2", role="assistant", text="Otto runs the office.",
                  drew={"shown": ["person:otto"]})
    remember_turn(store, thread="t2", role="assistant", text="Iris was mentioned in passing.",
                  drew={"read": ["person:iris"]})

    about = of_node(store, "person:iris")
    assert {t.text for t in about} == {"Iris surveys land.",
                                       "Iris was mentioned in passing."}

    # narrowed to what an answer was actually *about*, not what it opened on the way
    named = of_node(store, "person:iris", how=("shown",))
    assert [t.text for t in named] == ["Iris surveys land."]

    assert of_node(store, "topic:surveying", how=("shown",)) == []
    assert len(of_node(store, "topic:surveying")) == 1


def test_the_working_can_be_left_behind(store):
    """Showing or hiding the working is the caller's choice, and both are one query."""
    remember_turn(store, thread="t1", role="assistant", text="Iris does.",
                  drew={"shown": ["person:iris"], "read": ["topic:surveying"]})

    with_working = follow(store, "t1")[0]
    assert with_working.drew == {"shown": ["person:iris"], "read": ["topic:surveying"]}

    without = follow(store, "t1", working=False)[0]
    assert without.drew == {}
    assert without.text == "Iris does."          # the words are the same either way


def test_recent_is_what_goes_back_to_a_model(store):
    """Role and content only: sending the working back would spend context on a transcript
    of searching."""
    for n in range(8):
        remember_turn(store, thread="t1", role="user" if n % 2 == 0 else "assistant",
                      text=f"line {n}", drew={"read": ["person:iris"]})
    said = recent(store, "t1", turns=4)
    assert said == [{"role": "user" if n % 2 == 0 else "assistant", "content": f"line {n}"}
                    for n in (4, 5, 6, 7)]
    assert all(set(m) == {"role", "content"} for m in said)


def test_limit_reads_a_conversation_from_its_end(store):
    for n in range(5):
        remember_turn(store, thread="t1", role="user", text=f"line {n}")
    assert [t.text for t in follow(store, "t1", limit=2)] == ["line 3", "line 4"]


def test_threads_lists_what_is_held(store):
    remember_turn(store, thread="t1", role="user", text="one")
    remember_turn(store, thread="t1", role="user", text="two")
    remember_turn(store, thread="t2", role="user", text="three")

    held = {t["thread"]: t["turns"] for t in threads(store)}
    assert held == {"t1": 2, "t2": 1}


def test_forgetting_a_thread_takes_its_joins_with_it(store):
    remember_turn(store, thread="t1", role="assistant", text="Iris does.",
                  drew={"shown": ["person:iris"]})
    remember_turn(store, thread="t2", role="assistant", text="Otto does.",
                  drew={"shown": ["person:otto"]})

    assert forget_thread(store, "t1") == 1
    assert follow(store, "t1") == []
    assert of_node(store, "person:iris") == []          # the join went too
    assert len(follow(store, "t2")) == 1                # and the other thread is untouched
    assert forget_thread(store, "t1") == 0


def test_a_conversation_survives_being_reopened(tmp_path):
    """History that lives in a browser tab dies with the tab. This does not."""
    where = tmp_path / "graph.ladybug"
    with GraphStore(where) as store:
        store.write(GRAPH)
        remember_turn(store, thread="t1", role="user", text="who surveys land?")
        remember_turn(store, thread="t1", role="assistant", text="Iris does.",
                      drew={"shown": ["person:iris"]})

    with GraphStore(where, read_only=True) as reopened:
        said = follow(reopened, "t1")
        assert [t.text for t in said] == ["who surveys land?", "Iris does."]
        assert said[1].drew == {"shown": ["person:iris"]}
        assert [t.text for t in of_node(reopened, "person:iris")] == ["Iris does."]


def test_a_graph_that_was_never_talked_to_has_no_conversation(tmp_path):
    """Reading history from a fresh graph is empty, not an error.

    History is an addition to a graph, not part of every graph — and a reader cannot create
    the tables even if it wanted to. Found from the app side: asking a question without
    naming a thread left the tables unmade, and the next read raised "Table Turn does not
    exist" rather than saying there was nothing.
    """
    with GraphStore(tmp_path / "graph.ladybug") as store:
        store.write(GRAPH)
        assert follow(store, "never") == []
        assert threads(store) == []
        assert of_node(store, "person:iris") == []
        assert turn_of(store, "nosuchturn") is None
        assert recent(store, "never") == []
        assert forget_thread(store, "never") == 0

    # and read-only, where the tables certainly cannot be made
    with GraphStore(tmp_path / "graph.ladybug", read_only=True) as reopened:
        assert follow(reopened, "never") == []
        assert threads(reopened) == []
