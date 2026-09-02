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
        # the database breaks a tie between equal shortest paths its own way, and it broke
        # it differently on Linux (CI, 2026-09-02); what the question claims must be *a*
        # shortest path with the same ends, not the one this build happens to return
        assert found[0] == q["expect"][0] and found[-1] == q["expect"][-1], f"{q['q']}: {found}"
        assert len(found) == len(q["expect"]), f"{q['q']}: graph says {found}"
        ids = {n["id"] for n in graph()["nodes"]}
        assert set(q["expect"]) <= ids


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


def test_the_full_set_is_sixty_scored_questions_and_no_one_kind_of_them(g):
    """Sixty is the `n` a full run records -- the scored questions; the ones whose right
    answer is nobody are asked, not counted -- and the ranking takes a model's largest run,
    so a fifty-question row stays valid until a sixty of the same model exists.

    Each question is filed under the rarest kind it asks for, exactly as `bench.sample`
    files it when drawing a short run. Every bucket has to be there for a short run to
    have anything to draw, and none may be more than half the set: the set is about half
    person-shaped on purpose, because the page is, and half is the line past which it is
    person-shaped by accident again. Ids and duplicates are held by
    `test_every_expected_answer_exists` and `test_no_question_is_asked_twice`."""
    kind = {n["id"]: n["kind"] for n in g["nodes"]}
    scored = [q for q in QUESTIONS if q["expect"]]
    assert len(scored) == 60
    assert len(QUESTIONS) - len(scored) >= 4, "and some whose right answer is nobody"

    wanted = Counter(k for q in QUESTIONS
                     for k in ({kind[e] for e in q["expect"]} or {"nobody"}))
    filed = Counter(min({kind[e] for e in q["expect"]} or {"nobody"}, key=wanted.__getitem__)
                    for q in QUESTIONS)
    assert set(filed) == {"person", "org", "place", "topic", "opportunity", "event", "nobody"}
    assert all(n >= 2 for n in filed.values()), f"a kind with one question: {dict(filed)}"
    assert max(filed.values()) <= len(QUESTIONS) / 2, f"one kind is most of the set: {dict(filed)}"


# --- the four kinds the set was short of ------------------------------------------------------
#
# Counting, two hops, traps and quotes. Each is held here by deriving its answer from the graph
# again, so the expectation is a fact about the graph rather than something remembered.

_GAPS: dict[str, tuple[str, ...]] = {
    "aggregate": ("How many people here do robotics?",
                  "Which company sent the most people to the Northern Trade Fair?",
                  "Who here has been doing their job the longest?"),
    "two-hop": ("Who works alongside the person who does geotechnics?",
                "Which places do the people who do repair live in?"),
    "trap": ("Who here is called Vance?",
             "Since Ada Lovelace moved to Selby, who is left in Calderwick?",
             "Who here can weld, even a little?"),
    "quote": ("Who said they had just joined?",
              "What did Vera Lund say she works on?"),
}


def _asked(text: str) -> dict:
    return next(q for q in QUESTIONS if q["q"] == text)


def _who(g, rel: str, target: str) -> set[str]:
    return {e["source"] for e in g["edges"] if e["rel"] == rel and e["target"] == target}


def _of(g, source: str, rel: str) -> set[str]:
    return {e["target"] for e in g["edges"] if e["rel"] == rel and e["source"] == source}


def test_each_gap_has_at_least_two_scored_questions_whose_answers_exist(g):
    have = {n["id"] for n in g["nodes"]}
    asked = {q["q"] for q in QUESTIONS}
    for gap, texts in _GAPS.items():
        assert len(texts) >= 2, gap
        for text in texts:
            assert text in asked, text
            expect = _asked(text)["expect"]
            assert expect and set(expect) <= have, text


def test_a_count_is_scored_as_the_people_counted(g):
    assert set(_asked("How many people here do robotics?")["expect"]) == \
        _who(g, "experienced_in", "topic:robotics")


def test_the_company_that_sent_the_most_people_to_the_fair_is_the_unique_most(g):
    went = _who(g, "attended", "event:tradefair")
    sent = Counter(o for p in went for o in _of(g, p, "works_at"))
    top = sent.most_common()
    assert top[0][1] > top[1][1], f"a tie is not a comparative: {top}"
    assert _asked("Which company sent the most people to the Northern Trade Fair?")["expect"] \
        == [top[0][0]]


def test_the_longest_serving_person_is_the_only_one_who_said_the_largest_number_of_years(g):
    import re

    words = {"twelve": 12, "twenty": 20, "twenty-five": 25}
    years = {}
    for n in g["nodes"]:
        for m in n.get("messages") or ():
            for hit in re.findall(r"\b(twelve|twenty-five|twenty) years\b",
                                  g["messages"][m]["text"].lower()):
                years[n["id"]] = words[hit]
    assert len(years) >= 3, "several people put a number on it, or there is nothing to compare"
    most = max(years.values())
    assert [i for i, y in years.items() if y == most] == \
        _asked("Who here has been doing their job the longest?")["expect"]


def test_two_hops_answer_with_the_far_end_and_never_the_middle(g):
    tam = _who(g, "experienced_in", "topic:geotechnics")
    assert tam == {"person:tam"}
    colleagues = {p for org in _of(g, "person:tam", "works_at")
                  for p in _who(g, "works_at", org)} - tam
    assert set(_asked("Who works alongside the person who does geotechnics?")["expect"]) \
        == colleagues

    fixers = _who(g, "experienced_in", "topic:repair")
    homes = {place for p in fixers for place in _of(g, p, "based_in")}
    q = _asked("Which places do the people who do repair live in?")
    assert set(q["expect"]) == homes
    assert not (set(q["expect"]) & fixers)


def test_a_surname_alone_wants_everyone_who_carries_it(g):
    vances = {n["id"] for n in g["nodes"] if n["kind"] == "person" and n["label"].endswith(" Vance")}
    assert len(vances) == 2, "two of them, or there is nothing to confuse"
    assert set(_asked("Who here is called Vance?")["expect"]) == vances


def test_a_false_premise_leaves_the_place_exactly_as_the_graph_has_it(g):
    assert _of(g, "person:ada", "based_in") == {"place:turin"}, "Ada was never in Calderwick"
    trap = _asked("Since Ada Lovelace moved to Selby, who is left in Calderwick?")
    assert set(trap["expect"]) == _who(g, "based_in", "place:calderwick")
    assert trap["expect"] == _asked("Who is in Calderwick?")["expect"]


def test_a_near_miss_is_the_answer_only_when_the_question_allows_it(g):
    said = {n["id"]: " ".join(g["messages"][m]["text"] for m in n.get("messages") or ())
            for n in g["nodes"] if n["kind"] == "person"}
    welders = {i for i, s in said.items() if "weld" in s.lower()}
    assert "topic:welding" not in {n["id"] for n in g["nodes"]}, "nobody has it as a subject"
    assert _asked("Who here can weld, even a little?")["expect"] == sorted(welders)
    assert _asked("Nobody here does underwater welding. Who could?")["expect"] == []


def test_a_quote_question_is_answered_by_the_words_and_by_nothing_else(g):
    said = {n["id"]: " ".join(g["messages"][m]["text"] for m in n.get("messages") or ())
            for n in g["nodes"] if n["kind"] == "person"}
    joined = {i for i, s in said.items() if "just joined" in s.lower()}
    assert _asked("Who said they had just joined?")["expect"] == sorted(joined)
    assert not [e for e in g["edges"] if e["source"] == "person:pell"], "only the quote finds him"

    vera = said["person:vera"].lower()
    assert "data engineering" in vera and "hospital" in vera
    assert set(_asked("What did Vera Lund say she works on?")["expect"]) \
        == _of(g, "person:vera", "experienced_in")
