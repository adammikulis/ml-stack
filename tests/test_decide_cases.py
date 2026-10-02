"""The JSONL case format and the pointer prompt's option spans."""

from __future__ import annotations

import json

import pytest

from ml_stack.decide import pointer_prompt
from ml_stack.decide.cases import Case, case_from, fingerprint, read_cases, write_cases
from ml_stack.decide.types import options_of


def test_cases_round_trip_through_jsonl(tmp_path):
    cs = [Case("q1", "s1", options_of({"a": "first", "b": ""}), "b", id="1", group="g",
               tags=("t",)),
          Case("q2", {"k": [1, 2]}, options_of(["x", "y"]), "x", id="2")]
    path = tmp_path / "c.jsonl"
    assert write_cases(path, cs) == 2
    assert read_cases(path) == cs
    assert json.loads(path.read_text().splitlines()[0])["options"] == {"a": "first", "b": ""}


def test_a_label_that_is_not_an_option_is_refused():
    with pytest.raises(ValueError, match="not one of"):
        case_from({"question": "q", "state": "s", "options": ["a", "b"], "label": "c"})
    with pytest.raises(ValueError, match="no 'label'"):
        case_from({"question": "q", "state": "s", "options": ["a", "b"]})


def test_the_fingerprint_changes_with_any_case_and_with_order():
    a = Case("q", "s", options_of(["x", "y"]), "x")
    b = Case("q", "s", options_of(["x", "y"]), "y")
    assert fingerprint([a, b]) != fingerprint([b, a]) != fingerprint([a])
    assert fingerprint([a]) == fingerprint([Case("q", "s", options_of(["x", "y"]), "x")])


def test_each_option_line_is_a_span_of_the_prompt():
    opts = options_of({"billing": "payments  and\ninvoices", "sales": ""})
    r = pointer_prompt.render("Which team?", "Help!", opts)
    lines = [r.text[a:b] for a, b in r.spans]
    assert lines == ["1. billing — payments and invoices", "2. sales"]
    assert r.text.endswith("</options>\n</question>\n<answer>")
    assert r.text.startswith("<state>\nHelp!\n</state>\n")


def test_the_scoring_token_of_an_option_is_the_last_one_inside_its_line():
    offsets = [(0, 3), (3, 5), (6, 9), (9, 10), (11, 14)]
    assert pointer_prompt.positions(offsets, ((0, 5), (6, 10))) == [1, 3]
    with pytest.raises(ValueError, match="no token"):
        pointer_prompt.positions(offsets, ((20, 25),))
