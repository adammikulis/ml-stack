"""The logprob backend against a server that reports first-token log-probabilities."""

from __future__ import annotations

import json
import math

import pytest
from decide_fakes import logprob_handler

from poolhouse.decide import Calibration
from poolhouse.decide.logprob import Chat, LogprobDecider, letter_probabilities, render
from poolhouse.decide.types import DecideError, options_of

OPTIONS = {"safe": "reads only", "reversible": "can be undone", "destructive": "cannot be undone"}


def test_the_prompt_letters_the_options_and_keeps_the_state_in_tags():
    text = render("Is it destructive?", "rm -rf /", options_of(OPTIONS))
    assert "<state>\nrm -rf /\n</state>" in text
    assert "A. safe - reads only" in text and "C. destructive - cannot be undone" in text
    assert text.endswith("Answer with the letter only.")


def test_letter_mass_sums_spellings_and_ignores_other_tokens():
    top = [{"token": "A", "logprob": math.log(0.5)}, {"token": " A", "logprob": math.log(0.25)},
           {"token": "b", "logprob": math.log(0.125)}, {"token": "Hello", "logprob": -1.0},
           {"token": "Z", "logprob": -0.1}]
    assert letter_probabilities(top, 3) == pytest.approx([0.75, 0.125, 0.0])


def test_a_null_log_probability_counts_for_nothing_and_zero_counts_for_one():
    top = [{"token": "A", "logprob": 0.0}, {"token": "B", "logprob": None}]
    assert letter_probabilities(top, 2) == [1.0, 0.0]


def test_decide_normalises_over_the_letters_and_maps_them_to_option_names(server):
    seen: list[dict] = []
    fake = server(logprob_handler(lambda user: {"A": 0.1, "B": 0.2, "C": 0.6}, seen=seen))
    d = LogprobDecider(fake.base_url).decide("Is it destructive?", "x", OPTIONS)
    assert d.choice == "destructive"
    assert d.scores == pytest.approx({"safe": 0.1 / 0.9, "reversible": 0.2 / 0.9,
                                      "destructive": 0.6 / 0.9})
    assert d.backend == "logprob" and d.model == "qwen3-fake"
    assert d.details["letter_mass"] == pytest.approx(0.9)
    body = seen[0]
    assert (body["max_tokens"], body["temperature"], body["logprobs"]) == (1, 0, True)
    assert body["top_logprobs"] == 20


def test_the_request_switches_thinking_off_and_sends_the_token(server):
    seen: list[dict] = []
    fake = server(logprob_handler(lambda u: {"A": 0.5, "B": 0.5}, seen=seen))
    LogprobDecider(Chat(fake.base_url, key="s3cret")).decide("q", "s", ["x", "y"])
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    auth = [r for r in fake.requests if r[1].endswith("completions")]
    assert auth


def test_a_calibration_reshapes_the_scores(server):
    fake = server(logprob_handler(lambda user: {"A": 0.99, "B": 0.01}))
    plain = LogprobDecider(fake.base_url).decide("q", "s", ["x", "y"])
    flat = LogprobDecider(fake.base_url, calibration=Calibration(3.0)).decide("q", "s", ["x", "y"])
    assert flat.confidence < plain.confidence
    assert flat.choice == plain.choice == "x"


def test_a_model_that_answers_with_no_letter_is_an_error_not_a_guess(server):
    fake = server(logprob_handler(lambda user: {}))
    with pytest.raises(DecideError, match="option letters"):
        LogprobDecider(fake.base_url).decide("q", "s", ["x", "y"])
    garbled = server(lambda m, p, b: (200, json.dumps({"choices": [{"logprobs": {"content": [
        {"token": "@", "top_logprobs": [{"token": "#", "logprob": None}]}]}}]}).encode()))
    with pytest.raises(DecideError, match="'#'"):
        LogprobDecider(garbled.base_url).decide("q", "s", ["x", "y"])


def test_a_server_without_logprobs_is_named_in_the_error(server):
    fake = server(lambda m, p, b: (200, json.dumps({"choices": [{"message": {"content": "A"}}]}
                                                   ).encode()))
    with pytest.raises(DecideError, match="log-probabilities"):
        LogprobDecider(fake.base_url).decide("q", "s", ["x", "y"])


def test_more_options_than_letters_is_refused_before_any_request(server):
    fake = server(logprob_handler(lambda u: {"A": 1.0}))
    with pytest.raises(DecideError, match="at most 26"):
        LogprobDecider(fake.base_url).decide("q", "s", [f"o{i}" for i in range(27)])
    assert not [r for r in fake.requests if r[1].endswith("completions")]


def test_an_unreachable_server_is_a_decide_error():
    with pytest.raises(DecideError):
        LogprobDecider(Chat("http://127.0.0.1:9", timeout=1)).decide("q", "s", ["x", "y"])


def test_batch_and_async_forms_agree_with_the_single_call(server):
    import asyncio

    from poolhouse.decide.base import Request
    fake = server(logprob_handler(lambda user: {"A": 0.3, "B": 0.7}))
    d = LogprobDecider(fake.base_url)
    one = d.decide("q", "s", ["x", "y"])
    many = d.decide_many([Request("q", "s", ["x", "y"]), Request("q", "s", ["x", "y"])])
    later = asyncio.run(d.adecide("q", "s", ["x", "y"]))
    assert [m.scores for m in many] == [one.scores, one.scores] and later.scores == one.scores


def test_a_decision_stays_unthinking_whatever_the_person_set(server, monkeypatch):
    monkeypatch.setenv("POOLHOUSE_THINK", "on")
    seen: list[dict] = []
    fake = server(logprob_handler(lambda u: {"A": 0.5, "B": 0.5}, seen=seen))
    LogprobDecider(Chat(fake.base_url)).decide("q", "s", ["x", "y"])
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
