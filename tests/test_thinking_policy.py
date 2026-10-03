"""The per-use thinking policy: decisions and agent turns think only when the person says so."""

from __future__ import annotations

import pytest

from ml_stack.client import thinking


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv(thinking.ENV, raising=False)


def test_decisions_and_agent_turns_and_short_answers_default_to_off():
    assert [thinking.resolve(use) for use in (thinking.DECISION, thinking.AGENT, thinking.SHORT)] \
        == [False, False, False]


def test_the_person_turns_it_on_by_environment_or_argument_but_never_for_a_decision(monkeypatch):
    monkeypatch.setenv(thinking.ENV, "on")
    assert thinking.resolve(thinking.AGENT) and thinking.resolve(thinking.SHORT)
    assert not thinking.resolve(thinking.DECISION)
    assert not thinking.resolve(thinking.AGENT, asked="off")
    monkeypatch.delenv(thinking.ENV)
    assert thinking.resolve(thinking.SHORT, asked="on")


def test_auto_thinks_for_reasoning_or_when_the_prompt_asks_for_it():
    assert thinking.resolve(thinking.REASONING)
    assert thinking.resolve(thinking.SHORT, prompt="Please think step by step about this")
    assert not thinking.resolve(thinking.SHORT, prompt="What is the capital of France?")
    assert not thinking.resolve(thinking.SHORT, asked="off", prompt="think step by step")


def test_an_unknown_setting_is_auto(monkeypatch):
    monkeypatch.setenv(thinking.ENV, "maybe")
    assert thinking.setting() == "auto" and not thinking.resolve(thinking.AGENT)


def test_the_header_line_says_what_was_chosen_and_why(monkeypatch):
    assert thinking.describe(thinking.AGENT) == "thinking off (auto, agent)"
    assert thinking.describe(thinking.DECISION, asked="on") == "thinking off (always off, decision)"
    assert thinking.describe(thinking.SHORT, asked="on") == "thinking on (on, short)"


def test_the_policy_line_names_each_use():
    assert thinking.policy() == ("thinking auto: decision off, agent off, short off; "
                                 "on when the person or the task asks")
    assert "agent on" in thinking.policy("on") and "decision off" in thinking.policy("on")
