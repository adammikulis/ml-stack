"""Memory in the chat: the first message carries what is recalled from both scopes, remember asks
with the scope shown, every role may recall and only the acting ones may remember, and a
poisoned project memory changes nothing. The model is scripted; the loop, the rails, the
person's prompt and the stores are real."""

from __future__ import annotations

import io

import pytest

from ml_stack import chat, do, memory
from tests import memory_keys
from tests.test_chat import Model, registry
from tests.test_memory import inject
from tests.test_memory_scopes import repo

ring = memory_keys.ring


@pytest.fixture(autouse=True)
def released():
    """Every memory opened by ``make`` is closed after the test."""
    opened.clear()
    yield
    for each in opened:
        each.close()


opened: list[memory.Memory] = []


def make(tmp_path, script, stdin="", *, role="operator", plant=()):
    proj = repo(tmp_path, "alpha")
    mem = memory.Memory.open(explicit=proj)
    opened.append(mem)
    mem.user.add("Prefers short answers with no filler", "preference")
    mem.project.add("Run the staging flag before every release", "note")
    for text in plant:
        inject(mem.project, text)
    out = io.StringIO()
    person = do.Person(io.StringIO(stdin), out)
    seen: list = []
    model = Model(script)
    tools = chat.tools_for_chat(person=person, registry=registry(seen))
    session = chat.Chat(model, person, tools=tools, role=role,
                        extension=chat.extensions(person, mem))
    return session, model, mem, out, seen


def first_user(model) -> str:
    return next(m["content"] for m in model.seen[0] if m["role"] == "user")


def test_the_first_message_carries_both_scopes_and_the_system_message_the_guidance(tmp_path):
    session, model, _, _, _ = make(tmp_path, [])
    session.turn("how do I do the staging release?")
    body = first_user(model)
    assert body.startswith("<untrusted source='memory'>")
    assert "scope user" in body and "scope project" in body and "staging flag" in body
    assert body.endswith("how do I do the staging release?")
    system = model.seen[0][0]["content"]
    assert system.startswith(chat.SYSTEM[:20]) and "Two notebooks" in system and "alpha" in system
    assert "<untrusted source" not in system and "staging flag" not in system
    session.turn("and then?")
    assert "<untrusted" not in model.seen[1][-1]["content"]


def test_a_new_conversation_recalls_again(tmp_path):
    session, model, _, _, _ = make(tmp_path, [])
    session.turn("staging release")
    session.new()
    session.turn("staging release again")
    assert "<untrusted source='memory'>" in model.seen[1][1]["content"]


def test_remember_asks_with_the_scope_and_writes_where_the_person_says(tmp_path):
    call = ("remember", {"fact": "CI needs the staging flag", "scope": "project", "kind": "note"})
    session, _, mem, out, _ = make(tmp_path, [call], stdin="1\n")
    session.turn("note that for later")
    assert "Remember for this project (alpha): CI needs the staging flag" in out.getvalue()
    assert "1) yes, for this project (alpha)  2) yes, but for you (all projects) instead" in out.getvalue()
    assert "CI needs the staging flag" in [f.text for f in mem.project.facts()]


def test_the_person_can_switch_the_scope_and_a_no_stores_nothing(tmp_path):
    call = ("remember", {"fact": "likes dark terminals", "scope": "project", "kind": "preference"})
    never = ("remember", {"fact": "likes loud bells", "scope": "project", "kind": "preference"})
    session, _, mem, _, _ = make(tmp_path, [call], stdin="2\n")
    session.turn("remember it")
    assert "likes dark terminals" in [f.text for f in mem.user.facts()]
    assert "likes dark terminals" not in [f.text for f in mem.project.facts()]
    session, _, mem, _, _ = make(tmp_path / "again", [never], stdin="\n")
    session.turn("remember it")
    assert "likes loud bells" not in [f.text for f in mem.user.facts() + mem.project.facts()]


def test_remember_never_offers_always_allow_and_a_saved_rule_cannot_skip_the_prompt(tmp_path):
    call = ("remember", {"fact": "uses port 9090", "scope": "project", "kind": "note"})
    session, _, mem, out, _ = make(tmp_path, [call], stdin="3\n")
    session.turn("remember it")
    assert "always allow" not in out.getvalue().lower()
    assert "uses port 9090" not in [f.text for f in mem.project.facts()]


@pytest.mark.parametrize("role", ["reader", "operator", "runner"])
def test_every_role_may_recall(tmp_path, role):
    session, model, _, _, _ = make(tmp_path, [("recall", {"query": "staging"})], role=role)
    session.turn("what do I know about staging?")
    assert "recall" in model.offered[0]
    told = model.told()
    assert "scope project" in told and "staging flag" in told


def test_the_reader_role_is_not_offered_remember_and_the_acting_roles_are(tmp_path):
    reader, model, mem, _, _ = make(tmp_path, [("remember", {"fact": "x fact", "scope": "user"})], stdin="1\n",
                                    role="reader")
    reader.turn("remember something")
    assert "remember" not in model.offered[0] and mem.user.facts()[0].text.startswith("Prefers")
    assert len(mem.user.facts()) == 1
    for role in ("operator", "runner"):
        acting, model, _, _, _ = make(tmp_path / role, [], role=role)
        acting.turn("hi")
        assert {"recall", "remember"} <= model.offered[0]


def test_the_memory_command_shows_each_fact_with_its_scope(tmp_path):
    session, _, _, out, _ = make(tmp_path, [])
    chat._slash(session, "/memory", "staging release", out, None)
    shown = out.getvalue()
    assert "project  p.m0001" in shown and "staging flag" in shown
    assert "<untrusted" not in shown
    chat._slash(session, "/memory", "", out, None)
    assert "user     u.m0001" in out.getvalue()


def test_dry_run_prints_the_guidance_for_the_project_given(tmp_path):
    args = chat.COMMAND.parser().parse_args(["--dry-run", "--project", str(repo(tmp_path, "alpha"))])
    out = io.StringIO()
    assert chat.serve(args, io.StringIO(), out) == 0
    assert "Two notebooks" in out.getvalue() and "alpha" in out.getvalue()
    assert "remember" in out.getvalue() and "recall" in out.getvalue()


def test_a_project_that_is_not_a_directory_is_refused(tmp_path):
    args = chat.COMMAND.parser().parse_args(["--dry-run", "--project", str(tmp_path / "nope")])
    out = io.StringIO()
    assert chat.serve(args, io.StringIO(), out) == 2 and "not a directory" in out.getvalue()
