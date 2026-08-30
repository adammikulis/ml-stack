"""A question answered with ids. A model must never be able to name one that is not there."""

import pytest

from ml_stack.contracts import grammar_for
from ml_stack.entities import PICK_SCHEMA, pick, validate_pick
from ml_stack.entities.pick import objections

RECORDS = {"n:1": {"label": "Ada"}, "n:2": {"label": "Alan"}, "n:3": {"label": "looms"}}
IDS = list(RECORDS)


class Stub:
    def __init__(self, reply):
        self.reply = reply
        self.seen = []

    def extract(self, text, schema, **kw):
        self.seen.append((text, kw))
        return self.reply


def test_an_id_that_is_not_in_the_listing_is_dropped():
    got, why = validate_pick({"ids": ["n:1", "n:404"], "why": "both"}, ids=IDS)
    assert got == ["n:1"] and why == "both"


def test_the_order_asked_for_is_kept_and_repeats_collapse():
    got, _ = validate_pick({"ids": ["n:3", "n:1", "n:3"], "why": ""}, ids=IDS)
    assert got == ["n:3", "n:1"]


def test_a_limit_keeps_the_first():
    got, _ = validate_pick({"ids": ["n:3", "n:1", "n:2"], "why": ""}, ids=IDS, limit=2)
    assert got == ["n:3", "n:1"]


@pytest.mark.parametrize("raw", [None, [], "n:1", {}, {"ids": "n:1"}, {"ids": [1, None]}])
def test_anything_that_is_not_a_list_of_known_ids_gives_nothing(raw):
    assert validate_pick(raw, ids=IDS) == ([], "")


def test_a_non_string_why_is_not_passed_through():
    assert validate_pick({"ids": [], "why": 7}, ids=IDS) == ([], "")


def test_objections_name_every_invented_id():
    said = objections({"ids": ["n:1", "n:404", "n:405"]}, ids=IDS)
    assert len(said) == 2 and "n:404" in said[0]


def test_pick_asks_with_the_listing_and_keeps_only_what_checks_out():
    client = Stub({"ids": ["n:2", "nope"], "why": "he answered"})
    got, why = pick("who replied?", records=RECORDS, client=client)
    assert got == ["n:2"] and why == "he answered"
    text, kw = client.seen[0]
    assert "n:2" in text and "who replied?" in text
    assert kw["think"] is False and kw["schema_name"] == "graph_pick"


def test_pick_needs_neither_a_model_for_an_empty_question_nor_for_an_empty_graph():
    client = Stub({"ids": ["n:1"], "why": ""})
    assert pick("   ", records=RECORDS, client=client) == ([], "")
    assert pick("who?", records={}, client=client) == ([], "")
    assert client.seen == []


def test_the_schema_makes_a_grammar():
    assert grammar_for(PICK_SCHEMA) == grammar_for(PICK_SCHEMA)
    assert "ids" in grammar_for(PICK_SCHEMA)
