"""The invented community the benchmark asks about.

A question set rots quietly: a person is added, a second person acquires a subject, and an
expected answer that was right yesterday is half right today with nothing to say so. These
hold the shape of it. Every name here is invented.
"""

from __future__ import annotations

import pathlib
import tempfile
from collections import Counter

import pytest

from ml_stack.graph.community import QUESTIONS, _MORE_SAID, graph
from ml_stack.graph.store import GraphStore


@pytest.fixture(scope="module")
def g():
    return graph()


@pytest.fixture(scope="module")
def stored(g):
    with tempfile.TemporaryDirectory() as d:
        with GraphStore(pathlib.Path(d) / "g.ladybug") as store:
            store.write(g)
            yield store


def test_every_expected_answer_exists(g):
    """An expected id that is not in the graph can never be found, so the question is
    unanswerable and the score is a lie about the model."""
    have = {n["id"] for n in g["nodes"]}
    missing = sorted({e for q in QUESTIONS for e in q["expect"] if e not in have})
    assert missing == []


def test_the_graph_is_big_enough_to_have_to_discriminate(g):
    """A dozen entries make retrieval trivial: there is nothing to confuse. This number is
    a floor, not a target -- if it drops, the benchmark got easier without anyone saying so."""
    assert len(g["nodes"]) >= 100
    assert len(g["edges"]) >= 150
    kinds = Counter(n["kind"] for n in g["nodes"])
    # several people per subject, or "who knows X" has nothing to choose between
    assert kinds["person"] >= 40
    assert kinds["topic"] >= 20


def test_every_kind_the_page_draws_is_asked_about(g):
    """The page draws people, orgs, places, topics and opportunities. A benchmark that only
    asks about people measures a fifth of it -- and rewards anything that prefers people."""
    kind = {n["id"]: n["kind"] for n in g["nodes"]}
    asked = {kind[e] for q in QUESTIONS for e in q["expect"]}
    assert {"person", "org", "place", "topic", "opportunity", "event"} <= asked


def test_a_useful_share_of_questions_want_no_person_at_all(g):
    """The set was nine-tenths person-shaped, which flattered a filter that prefers people.
    A rule can only be measured against questions that could falsify it."""
    kind = {n["id"]: n["kind"] for n in g["nodes"]}
    scored = [q for q in QUESTIONS if q["expect"]]
    peopleless = [q for q in scored
                  if not any(kind[e] == "person" for e in q["expect"])]
    assert len(peopleless) / len(scored) >= 0.20


def test_the_crowd_is_never_an_answer(g):
    """The crowd exists to be plausible and wrong. If one of them becomes a right answer,
    adding more crowd silently changes every score."""
    crowd = set(_MORE_SAID)
    assert crowd, "there should be a crowd"
    used = sorted({e for q in QUESTIONS for e in q["expect"] if e in crowd})
    assert used == []


def test_the_crowd_is_near_enough_to_be_confusing(g):
    """Distractors that share no vocabulary with the questions are not distractors: a search
    never surfaces them and they cost the model nothing to ignore."""
    crowd_people = [i for i in _MORE_SAID if i.startswith("person:")]
    assert len(crowd_people) >= 30
    said = " ".join(str((g["messages"].get(m) or {}).get("text", ""))
                    for n in g["nodes"] for m in (n.get("messages") or ()))
    # the crowd talks about work, in the same register as the cast
    for near in ("hydraulics", "procurement", "calibration"):
        assert near in said


def test_the_paths_the_questions_claim_are_the_paths_the_graph_has(stored):
    """A pathing question's answer is derivable, so it is derived rather than remembered."""
    for q in QUESTIONS:
        if not q["q"].lower().startswith(("how are", "how is", "what links")):
            continue
        people = [e for e in q["expect"] if e.startswith("person:")]
        if len(people) < 2:
            continue                      # "no path" questions expect nothing
        found = stored.shortest_path(people[0], people[-1])
        assert found == q["expect"], f"{q['q']}: graph says {found}"


def test_a_pair_with_no_path_really_has_none(stored):
    """"How is X connected to Y" with an empty answer must be a graph fact, not an omission."""
    assert stored.shortest_path("person:iris", "person:alan") == []


def test_the_graph_is_the_same_every_time():
    """The crowd is generated. A benchmark whose graph moves between runs measures nothing."""
    once, twice = graph(), graph()
    assert [n["id"] for n in once["nodes"]] == [n["id"] for n in twice["nodes"]]
    assert once["edges"] == twice["edges"]
    assert once["messages"] == twice["messages"]


def test_nobody_is_joined_to_something_that_does_not_exist(g):
    have = {n["id"] for n in g["nodes"]}
    dangling = sorted({e["source"] for e in g["edges"] if e["source"] not in have}
                      | {e["target"] for e in g["edges"] if e["target"] not in have})
    assert dangling == []


def test_no_question_is_asked_twice():
    """A duplicate scores twice and, if its two copies disagree about the answer, scores
    itself both right and wrong. One slipped in during an edit and only the path check saw
    it -- because the two copies had the same words and different expectations."""
    from collections import Counter

    said = Counter(q["q"].strip().casefold() for q in QUESTIONS)
    assert [q for q, n in said.items() if n > 1] == []


def test_the_full_set_is_fifty_scored_questions_and_no_one_kind_of_them(g):
    """Fifty is the `n` a full run records -- the scored questions; the ones whose right
    answer is nobody are asked, not counted -- and the ranking takes a model's largest run,
    so a thirty-four-question row stays valid until a fifty of the same model exists.

    Each question is filed under the rarest kind it asks for, exactly as `bench.sample`
    files it when drawing a short run. Every bucket has to be there for a short run to
    have anything to draw, and none may be more than half the set: the set is about half
    person-shaped on purpose, because the page is, and half is the line past which it is
    person-shaped by accident again. Ids and duplicates are held by
    `test_every_expected_answer_exists` and `test_no_question_is_asked_twice`."""
    kind = {n["id"]: n["kind"] for n in g["nodes"]}
    scored = [q for q in QUESTIONS if q["expect"]]
    assert len(scored) == 50
    assert len(QUESTIONS) - len(scored) >= 4, "and some whose right answer is nobody"

    wanted = Counter(k for q in QUESTIONS
                     for k in ({kind[e] for e in q["expect"]} or {"nobody"}))
    filed = Counter(min({kind[e] for e in q["expect"]} or {"nobody"}, key=wanted.__getitem__)
                    for q in QUESTIONS)
    assert set(filed) == {"person", "org", "place", "topic", "opportunity", "event", "nobody"}
    assert all(n >= 2 for n in filed.values()), f"a kind with one question: {dict(filed)}"
    assert max(filed.values()) <= len(QUESTIONS) / 2, f"one kind is most of the set: {dict(filed)}"
