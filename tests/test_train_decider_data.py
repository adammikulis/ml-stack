"""Guard cases synthesised from tool schemas, and the registry of trained deciders."""

from __future__ import annotations

import collections
import json

import pytest

from poolhouse.decide import registry
from poolhouse.decide.guards.scoperules import ScopeDecider, violations
from poolhouse.decide.sources import CONFIG, FORMAT
from poolhouse.decide.types import DecideError
from poolhouse.train.decider_data import Plan, default_prompts, guard_cases_from_tools, risk_of


def tool(name, description, **props):
    return {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {k: {"type": "string", "description": v}
                                         for k, v in props.items()},
        "required": list(props)}}


TOOLS = [
    tool("read_page", "Read a wiki page.", path='page path e.g. "docs/intro.md"'),
    tool("delete_page", "Delete a page permanently.", path='page path e.g. "docs/old.md"'),
    tool("create_note", "Create a note.", title='e.g. "standup"'),
    tool("fetch_url", "Fetch a URL.", url='e.g. "https://docs.example.test/a"'),
    tool("send_message", "Send a message to a channel.", channel='e.g. "#team"', text="the text"),
]


def test_risk_comes_from_an_override_then_the_name_then_the_description():
    schemas = {t["name"]: {"function": t} for t in TOOLS}
    assert risk_of(schemas["delete_page"]) == "destructive"
    assert risk_of(schemas["create_note"]) == "reversible"
    assert risk_of(schemas["read_page"]) == "safe"
    assert risk_of(schemas["read_page"], {"read_page": "destructive"}) == "destructive"
    with pytest.raises(ValueError, match="one of"):
        risk_of(schemas["read_page"], {"read_page": "bad"})


def test_cases_cover_all_three_questions_with_both_sides_of_each():
    cases = guard_cases_from_tools(TOOLS, Plan(seed=1, per_tool=5))
    seen = collections.Counter((c.tags[0], c.label) for c in cases)
    assert {("destructive", "safe"), ("destructive", "destructive"), ("grounded", "grounded"),
            ("grounded", "injected"), ("scope", "inside"), ("scope", "outside")} <= set(seen)
    assert all("synthetic" in c.tags for c in cases)
    assert len({c.id for c in cases}) == len(cases)


def test_an_injected_case_calls_a_tool_the_request_did_not_ask_for():
    cases = guard_cases_from_tools(TOOLS, Plan(seed=1, per_tool=5))
    for c in (c for c in cases if c.label == "injected"):
        request = c.state.split("\n")[0]
        called = c.state.rsplit("Tool call: ", 1)[1].split("(")[0]
        assert called in c.state.split("Recent tool output")[1].split("Tool call:")[0]
        assert called not in request


def test_scope_labels_agree_with_the_deterministic_check():
    cases = guard_cases_from_tools(TOOLS, Plan(seed=2, per_tool=6))
    rules = ScopeDecider()
    scope = [c for c in cases if c.tags[0] == "scope"]
    assert scope and all(rules.decide(c.question, c.state, c.options).choice == c.label
                         for c in scope)


def test_the_same_seed_gives_the_same_cases_and_another_seed_different_ones():
    a = guard_cases_from_tools(TOOLS, Plan(seed=3))
    assert [c.state for c in a] == [c.state for c in guard_cases_from_tools(TOOLS, Plan(seed=3))]
    assert [c.state for c in a] != [c.state for c in guard_cases_from_tools(TOOLS, Plan(seed=4))]


def test_no_calls_is_an_error_and_a_served_model_can_supply_questions():
    with pytest.raises(ValueError):
        guard_cases_from_tools(TOOLS, Plan(prompts={"read_page": []}, per_tool=0))
    asked: list[str] = []

    def ask(prompt):
        asked.append(prompt)
        return '{"question": "show me the intro page", "arguments": {"path": "docs/intro.md"}}'

    cases = guard_cases_from_tools(TOOLS[:1], Plan(ask=ask, per_tool=3))
    assert asked and any("show me the intro page" in c.state for c in cases)


def test_the_default_questions_are_the_first_sentence_reworded():
    got = default_prompts([{"function": TOOLS[1]}])
    assert got == {"delete_page": ["Delete a page permanently.",
                                   "Please delete a page permanently.",
                                   "Can you delete a page permanently?"]}


def test_violations_ignore_the_project_for_tools_with_no_paths_or_urls():
    assert violations({"title": "standup"}, "/work/app", ()) == []


def make_dir(root, name):
    root.mkdir(parents=True)
    (root / CONFIG).write_text(json.dumps({"format": FORMAT, "name": name,
                                           "base": {"repo": "b/m"}}))
    (root / "manifest.json").write_text(json.dumps({"data_hash": "abc", "metrics": {"n": 3}}))
    return root


def test_registering_records_the_directory_under_its_name_and_replaces_a_repeat(tmp_path):
    first = make_dir(tmp_path / "one", "guard")
    registry.register(first)
    assert registry.find("guard") == first.resolve()
    again = make_dir(tmp_path / "two", "guard")
    with pytest.raises(DecideError, match="already registered"):
        registry.register(again)
    registry.register(again, replace=True)
    assert [r["path"] for r in registry.listing() if r["name"] == "guard"] == [str(again.resolve())]
    assert registry.find("guard.prev") == first.resolve()


def test_a_registered_directory_that_is_gone_is_not_listed(tmp_path):
    root = make_dir(tmp_path / "gone", "tmp")
    registry.register(root)
    (root / CONFIG).unlink()
    assert registry.listing() == []
    with pytest.raises(DecideError, match="no decider called"):
        registry.find("tmp")


def test_a_directory_that_is_not_a_decider_is_not_registered(tmp_path):
    root = tmp_path / "x"
    root.mkdir()
    (root / CONFIG).write_text(json.dumps({"format": "other"}))
    with pytest.raises(DecideError):
        registry.register(root)
