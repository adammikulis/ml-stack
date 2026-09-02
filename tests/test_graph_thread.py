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


# ------------------------------------------------------------ of any length

from ml_stack.graph.ask import EARLIER, RECALLED, converse  # noqa: E402
from ml_stack.graph.thread import (EVERY, SUMMARY, WINDOW, latest_summary,  # noqa: E402
                                   recall, summarise, write_summary)
from ml_stack.testing import ScriptedModel  # noqa: E402

LONG_GRAPH = {
    "nodes": [*GRAPH["nodes"],
              {"id": "person:marlow", "label": "Marlow Fen", "kind": "person",
               "attrs": {}, "messages": []},
              {"id": "place:calder", "label": "Calder Ridge", "kind": "place",
               "attrs": {}, "messages": []}],
    "edges": list(GRAPH["edges"]),
    "messages": {},
}

FACT = "Marlow Fen has moved to Calder Ridge to run the survey office there."
NOTED = "Noted: Marlow Fen is now at Calder Ridge."
QUESTION = "where did Marlow Fen move to?"


def bag(texts):
    """A stand-in embedder: a hashed bag of words, so meaning is word overlap and nothing
    talks to a model. Deterministic across processes."""
    import math
    out = []
    for text in texts:
        vector = [0.0] * 64
        for word in text.casefold().split():
            word = word.strip("?.,:;!")
            if word:
                vector[sum(ord(c) * (i + 1) for i, c in enumerate(word)) % 64] += 1.0
        size = math.sqrt(sum(x * x for x in vector)) or 1.0
        out.append([x / size for x in vector])
    return out


def notes(turns):
    """A scripted summary writer: keeps what was established and every id it is given."""
    kept = [t.text.split(" Rests on: ")[0] for t in turns if t.role == SUMMARY]
    facts = [t.text for t in turns if t.role != SUMMARY and "Marlow" in t.text]
    ids = dict.fromkeys(i for t in turns for how in t.drew for i in t.drew[how])
    established = " ".join(kept + facts) or "Nothing established yet."
    return f"{established} Rests on: {', '.join(ids)}."


@pytest.fixture(scope="module")
def long_thread(tmp_path_factory):
    """Two hundred turns: the first states a fact, the rest are noise about other entries,
    and the two hundredth asks about the fact. Built once; every test reads it read-only.

    ``prefixes`` is the summary the ask path would have sent at each of turns 193-200,
    taken as each turn was about to be asked -- the state at the time, not reconstructed.
    """
    where = tmp_path_factory.mktemp("long") / "graph.ladybug"
    prefixes = []
    with GraphStore(where) as store:
        store.write(LONG_GRAPH)
        remember_turn(store, thread="long", role="user", text=FACT, embedder=bag)
        remember_turn(store, thread="long", role="assistant", text=NOTED,
                      drew={"shown": ["person:marlow", "place:calder"]}, embedder=bag)
        summarise(store, "long", notes)
        for n in range(3, 201):
            if n >= 193:
                prefixes.append(latest_summary(store, "long").text)
            if n == 200:
                break                  # the question is asked, not yet remembered
            if n % 2:
                text = f"tell me about surveying instrument number {n}"
                drew = None
            else:
                text = f"Instrument number {n - 1} is a theodolite Iris keeps."
                drew = {"read": ["topic:surveying"]}
            remember_turn(store, thread="long", role="user" if n % 2 else "assistant",
                          text=text, drew=drew, embedder=bag)
            summarise(store, "long", notes)
    return where, prefixes


def test_recall_finds_the_fact_from_turn_one_two_hundred_turns_later(long_thread):
    """The fact left the window 190 turns ago; the word index and the vectors bring it back."""
    where, _ = long_thread
    with GraphStore(where, read_only=True) as store:
        said = follow(store, "long")
        assert len(said) == 199 and said[0].text == FACT
        cut = said[-WINDOW].seq

        by_words = recall(store, "long", QUESTION)
        assert [t.text for t in by_words][:2] == [FACT, NOTED]
        assert all(t.seq < cut and t.role != SUMMARY for t in by_words)
        assert by_words[1].drew == {"shown": ["person:marlow", "place:calder"]}

        fused = recall(store, "long", QUESTION, embedder=bag)
        assert FACT in [t.text for t in fused]
        assert all(t.seq < cut and t.role != SUMMARY for t in fused)
        assert [t.seq for t in fused] == sorted(t.seq for t in fused)   # oldest first

        # a question about nothing in particular still never reaches into the window
        assert all(t.seq < cut for t in recall(store, "long", "anything at all?", embedder=bag))
        assert recall(store, "long", "   ") == []
        assert recall(store, "long", QUESTION, limit=0) == []


def test_the_summary_still_carries_the_facts_id_two_hundred_turns_later(long_thread):
    """Rolled forward every EVERY turns, the summary keeps the id it was given each time."""
    where, _ = long_thread
    with GraphStore(where, read_only=True) as store:
        summary = latest_summary(store, "long")
        assert summary is not None and summary.role == SUMMARY
        assert FACT in summary.text and "person:marlow" in summary.text
        assert set(summary.drew["shown"]) >= {"person:marlow", "place:calder", "topic:surveying"}
        # 199 ordinary turns: a summary after every EVERY of them, none for the last seven
        everything = follow(store, "long", summaries=True)
        written = [t for t in everything if t.role == SUMMARY]
        assert len(written) == 199 // EVERY == 24
        assert written[-1].id == summary.id
        before = everything[[t.id for t in everything].index(summary.id) - 1]
        assert summary.meta["over"][1] == before.seq and before.role == "assistant"


def test_follow_keeps_summaries_out_of_the_window(long_thread):
    where, _ = long_thread
    with GraphStore(where, read_only=True) as store:
        window = recent(store, "long", turns=WINDOW)
        assert len(window) == WINDOW
        assert [m["content"] for m in window] == [t.text for t in follow(store, "long")[-WINDOW:]]
        assert all(m["role"] in ("user", "assistant") for m in window)
        assert all(t.role != SUMMARY for t in follow(store, "long"))
        assert any(t.role == SUMMARY for t in follow(store, "long", summaries=True))
        # in order either way
        seqs = [t.seq for t in follow(store, "long", summaries=True)]
        assert seqs == sorted(seqs)


def test_turn_two_hundred_is_assembled_summary_then_recalled_then_the_window(long_thread):
    """What the model sees at turn 200: the summary first, the recalled turn, only the last
    WINDOW ordinary turns, then the question -- and the fact is in front of it twice."""
    where, _ = long_thread
    with GraphStore(where, read_only=True) as store:
        window = recent(store, "long", turns=WINDOW)
        summary = latest_summary(store, "long")
        recalled = recall(store, "long", QUESTION, embedder=bag)
    model = ScriptedModel([], answer="Calder Ridge.")
    converse(QUESTION, LONG_GRAPH, model, turns=window, summary=summary, recalled=recalled)

    seen = model.seen[0]
    assert seen[0]["role"] == "system"
    assert seen[1] == {"role": "user", "content": EARLIER + summary.text}
    brought = seen[2:2 + len(recalled)]
    assert [m["content"] for m in brought] == [RECALLED + t.text for t in recalled]
    assert [m["role"] for m in brought] == [t.role for t in recalled]
    assert seen[2 + len(recalled):-1] == window
    assert seen[-1] == {"role": "user", "content": QUESTION}
    assert len(seen) == 1 + 1 + len(recalled) + WINDOW + 1
    assert sum(FACT in m["content"] for m in seen) == 2      # the summary and the recall


def test_the_prefix_is_the_same_across_the_last_eight_turns(long_thread):
    """System plus summary is identical from turn 193 to 200, so the cached prefix holds:
    the summary was last rolled at turn 192 and nothing per question sits ahead of it."""
    where, prefixes = long_thread
    assert len(prefixes) == 8 and len(set(prefixes)) == 1
    with GraphStore(where, read_only=True) as store:
        assert prefixes[0] == latest_summary(store, "long").text
    heads = []
    for n, text in enumerate(prefixes):
        model = ScriptedModel([])
        converse(f"question at turn {193 + n}", LONG_GRAPH, model, summary=text,
                 recalled=[{"role": "user", "content": f"recalled for {n}"}],
                 turns=[{"role": "user", "content": f"window {n}"}])
        heads.append(model.seen[0][:2])
    assert all(head == heads[0] for head in heads)
    assert heads[0][1]["content"].startswith(EARLIER)


def test_a_summary_is_written_only_when_enough_has_been_said(store):
    """Under EVERY new turns is not yet time; the writer is not called; nothing is kept."""
    calls = []

    def writer(turns):
        calls.append(list(turns))
        return "Iris surveys land; rests on person:iris and nothing else."

    for n in range(EVERY - 1):
        remember_turn(store, thread="t1", role="user", text=f"line {n}",
                      drew={"read": ["person:iris", "topic:surveying"]})
        assert summarise(store, "t1", writer) is None
    assert calls == [] and latest_summary(store, "t1") is None

    remember_turn(store, thread="t1", role="assistant", text="Iris does.",
                  drew={"shown": ["person:iris"]})
    summary = summarise(store, "t1", writer)
    assert summary is not None and summary.role == SUMMARY
    assert len(calls) == 1 and [t.role for t in calls[0]] == ["user"] * (EVERY - 1) + ["assistant"]
    # joined only to the ids it names that a summarised turn drew on: the writer dropped
    # topic:surveying, and "nothing else" is not an id
    assert summary.drew == {"shown": ["person:iris"]}
    assert summary.meta["over"] == [1, EVERY]
    assert latest_summary(store, "t1").id == summary.id
    assert [t.role for t in follow(store, "t1")] == ["user"] * (EVERY - 1) + ["assistant"]
    assert follow(store, "t1", summaries=True)[-1].role == SUMMARY

    # the next one is handed the previous summary first, and a writer that says nothing
    # writes nothing
    for n in range(EVERY):
        remember_turn(store, thread="t1", role="user", text=f"more {n}")
    quiet = summarise(store, "t1", lambda turns: (calls.append(list(turns)), "")[1])
    assert quiet is None and calls[1][0].id == summary.id
    assert [t.text for t in calls[1][1:]] == [f"more {n}" for n in range(EVERY)]
    assert latest_summary(store, "t1").id == summary.id


def test_write_summary_asks_a_client_once_with_the_turns_and_their_ids(store):
    turns = [remember_turn(store, thread="t1", role=SUMMARY, text="Earlier notes.",
                           drew={"shown": ["person:otto"]}),
             remember_turn(store, thread="t1", role="user", text="who surveys?"),
             remember_turn(store, thread="t1", role="assistant", text="Iris does.",
                           drew={"shown": ["person:iris"], "read": ["topic:surveying"]})]
    model = ScriptedModel([], answer="  Iris surveys land; rests on person:iris.  ")
    assert write_summary(model, turns) == "Iris surveys land; rests on person:iris."
    assert len(model.seen) == 1
    system, asked = model.seen[0]
    assert system["role"] == "system" and "one paragraph" in system["content"]
    assert asked["content"].splitlines() == [
        "previous notes: Earlier notes.  [rests on: person:otto]",
        "user: who surveys?",
        "assistant: Iris does.  [rests on: topic:surveying, person:iris]"]


def test_recall_never_returns_a_turn_inside_the_window(store):
    """Every turn matches; only the ones the window is not already sending come back."""
    for n in range(12):
        remember_turn(store, thread="t1", role="user", text=f"Iris and surveying, take {n}")
    got = recall(store, "t1", "what about Iris and surveying?", window=10, limit=5)
    assert {t.seq for t in got} <= {1, 2} and got
    assert recall(store, "t1", "Iris", window=12) == []
    whole = recall(store, "t1", "Iris surveying", window=0, limit=3)
    assert len(whole) == 3 and [t.seq for t in whole] == sorted(t.seq for t in whole)


def test_forgetting_a_thread_takes_its_vectors_too(store):
    remember_turn(store, thread="t1", role="user", text="Iris surveys", embedder=bag)
    remember_turn(store, thread="t2", role="user", text="Otto runs the office", embedder=bag)
    assert store.similar(bag(["Iris surveys"])[0], model="thread:t1")
    assert forget_thread(store, "t1") == 1
    assert store.similar(bag(["Iris surveys"])[0], model="thread:t1") == []
    assert store.similar(bag(["Otto runs the office"])[0], model="thread:t2")


def test_a_graph_never_talked_to_has_nothing_to_recall_or_summarise(tmp_path):
    with GraphStore(tmp_path / "graph.ladybug") as store:
        store.write(GRAPH)
        assert recall(store, "never", "who?") == []
        assert latest_summary(store, "never") is None
        assert summarise(store, "never", lambda turns: "notes") is None
    with GraphStore(tmp_path / "graph.ladybug", read_only=True) as reopened:
        assert recall(reopened, "never", "who?", embedder=bag) == []
        assert latest_summary(reopened, "never") is None
