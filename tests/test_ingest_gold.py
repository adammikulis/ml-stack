"""The shipped extraction gold set: twenty invented passages with everything they state
written down, scored through `gold_score` with scripted readings."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from test_ingest import EMPTY, a_model

from ml_stack import ingest

pytest.importorskip("pymupdf")

GOLD = Path(__file__).parent / "fixtures" / "extraction-gold.json"


@pytest.fixture(scope="module")
def passages():
    return ingest.read_gold(GOLD)


def _passage_for(prompt, passages):
    return next(p for p in passages if p["text"] in prompt)


def _as_said(triple):
    return {"from": triple["subject"], "rel": triple["predicate"], "to": triple["object"]}


def _flipped(triple):
    """The triple the other way round, through a verb whose inverse names its predicate."""
    for verb, other_way in ingest.INVERSES.items():
        if triple["predicate"] in other_way:
            return {"from": triple["object"], "rel": verb, "to": triple["subject"]}
    return None


def _score(server, passages, script):
    from ml_stack.client import Client

    instance, asked = a_model(server, script)
    scored = ingest.gold_score(Client(instance.base_url), passages, ingest.schema())
    return scored, asked


def test_the_set_holds_twenty_passages_each_stating_at_least_three_triples(passages):
    assert len(passages) == 20
    ids = [p["passage_id"] for p in passages]
    assert len(set(ids)) == 20
    for p in passages:
        assert p["source"] and p["text"]
        assert len(p["triples"]) >= 3, p["passage_id"]
        for triple in p["triples"]:
            assert triple["subject"] and triple["predicate"] and triple["object"]


def test_every_predicate_is_a_word_the_schema_has(passages):
    words = ingest.vocabulary(ingest.schema())
    assert words == set(ingest.VERBS)
    for p in passages:
        for triple in p["triples"]:
            assert ingest.sayable(triple, words), f"{p['passage_id']}: {triple['predicate']}"
            assert triple["predicate"] in words, f"{p['passage_id']}: {triple['predicate']}"
    assert not ingest.sayable({"predicate": "drains"}, words)
    assert ingest.sayable({"predicate": "caused_by"}, words), "a flip of `causes`"
    assert ingest.sayable({"predicate": "drains"}, set()), "no vocabulary says everything"


def test_every_inverse_verb_is_stated_at_least_twice(passages):
    counts = Counter(t["predicate"] for p in passages for t in p["triples"])
    for verb in ingest.INVERSES:
        assert counts[verb] >= 2, verb
    assert set(counts) == set(ingest.VERBS)


def test_a_reading_that_says_exactly_the_gold_scores_one_on_both_rates(server, passages):
    def script(prompt):
        return dict(EMPTY, relations=[_as_said(t) for t in _passage_for(prompt, passages)["triples"]])

    scored, asked = _score(server, passages, script)

    assert len(asked) == 20
    assert scored.wanted == scored.found == scored.matched == sum(len(p["triples"]) for p in passages)
    assert scored.recall == 1.0 and scored.precision == 1.0
    assert scored.misses == [] and scored.spurious == [] and scored.unsayable == []


def test_one_triple_the_passage_does_not_state_costs_precision_and_not_recall(server, passages):
    extra = {"from": "the reader", "rel": "member_of", "to": "the audience"}

    def script(prompt):
        return dict(EMPTY, relations=[_as_said(t) for t in _passage_for(prompt, passages)["triples"]]
                    + [extra])

    scored, _ = _score(server, passages, script)

    assert scored.recall == 1.0
    assert scored.precision == round(scored.wanted / (scored.wanted + 20), 4) < 1.0
    assert len(scored.spurious) == 20
    assert {m["triple"] for m in scored.spurious} == {"the reader member_of the audience"}


def test_a_reading_that_says_the_flippable_triples_the_other_way_round_still_scores_full_recall(
        server, passages):
    flipped = 0

    def script(prompt):
        nonlocal flipped
        said = []
        for triple in _passage_for(prompt, passages)["triples"]:
            other = _flipped(triple)
            flipped += other is not None
            said.append(other or _as_said(triple))
        return dict(EMPTY, relations=said)

    scored, _ = _score(server, passages, script)

    assert flipped >= 20, "part_of and has_part go both ways"
    assert scored.recall == 1.0 and scored.precision == 1.0
    assert scored.misses == []


def test_the_shipped_file_is_the_shape_read_gold_documents():
    held = json.loads(GOLD.read_text(encoding="utf-8"))
    assert set(held) == {"name", "passages"}
    keys = {"passage_id", "source", "text", "triples"}
    optional = {"subject_aliases", "predicate_aliases", "object_aliases"}
    for p in held["passages"]:
        assert set(p) == keys
        assert 3 <= p["text"].count(". ") + 1 <= 6, p["passage_id"]
        for t in p["triples"]:
            assert {"subject", "predicate", "object"} <= set(t) <= {"subject", "predicate", "object", *optional}
