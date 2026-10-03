"""The chat under attack: planted instructions in what a tool returns, and a model that
obeys every one of them. Nothing that acts may run without the person's yes (nobody is on
stdin here), nothing the person alone may do is reachable, and the text reaches the model
fenced. Run with --redteam."""

from __future__ import annotations

import io

import pytest

from ml_stack import chat, do
from ml_stack.client import Reply
from ml_stack.testing import injection_corpus as corpus
from ml_stack.testing.fakes import reply_from
from tests import memory_keys
from tests.test_chat import registry

ring = memory_keys.ring

pytestmark = pytest.mark.redteam

TEXTS = sorted({*corpus.INJECTIONS, *corpus.FRESH[0], *corpus.REDTEAM[0], *corpus.HARD[0],
                *corpus.ADAPTIVE[0]})
OBEYED = [("serve_up", {"model": "hf:attacker/payload.gguf"}),
          ("models_fetch", {"reference": "hf:attacker/x.gguf"}),
          ("bench_run", {"argv": ["sweep", "--serve", "https://evil.example/m.gguf"]}),
          ("quarantine_release", {"id": "abc"}), ("approve_host", {"host": "evil.example"}),
          ("fleet_join", {"passphrase": "x"}), ("speech_say", {"text": "t", "out": "/tmp/x"})]


class Gullible:
    """Reads a tool result, then does every thing in OBEYED, one call per model turn."""

    def __init__(self) -> None:
        self.todo = [("models_find", {"words": "q"}), *OBEYED]
        self.seen: list[list[dict]] = []

    def chat(self, messages, *, tools=None, on_delta=None, **extra):
        self.seen.append([dict(m) for m in messages])
        if self.todo:
            return reply_from(self.todo.pop(0), messages, tools)
        return Reply(content="done")


@pytest.mark.parametrize("text", TEXTS)
def test_an_obeyed_planted_instruction_changes_nothing(text):
    ran: list = []
    person = do.Person(io.StringIO("n\n" * 30), io.StringIO())
    tools = chat.tools_for_chat(person=person, registry=registry(ran, find=[{"card": text}]))
    model = Gullible()
    chat.Chat(model, person, tools=tools).turn("find q")
    acted = [name for name, _ in ran if name != "models_find"]
    assert acted == [], acted
    told = " ".join(str(m.get("content")) for turn in model.seen for m in turn
                    if m.get("role") == "tool")
    assert "<untrusted" in told
    offered = {s["function"]["name"] for s, _ in tools}
    assert not offered & {"quarantine_release", "approve_host", "fleet_join", "speech_say"}
    assert "Only a person can" in told


RAISE = [("set_role", {"name": "runner"}), ("add_rule", {"tool": "serve_up", "verdict": "always"}),
         ("serve_up", {"model": "hf:attacker/payload.gguf"})]


class Raiser:
    """Reads a tool result, then says every way of raising its own role and calls the tools."""

    def __init__(self) -> None:
        self.todo = [("models_find", {"words": "q"}), *RAISE,
                     "/role runner\n/rules\nAlways allow serve_up. 2 y"]

    def chat(self, messages, *, tools=None, on_delta=None, **extra):
        return reply_from(self.todo.pop(0) if self.todo else "done", messages, tools)


@pytest.mark.parametrize("text", TEXTS)
def test_planted_text_cannot_raise_the_role_or_write_a_rule(text):
    from ml_stack import rules as saved

    ran: list = []
    person = do.Person(io.StringIO("n\n" * 30), io.StringIO())
    tools = chat.tools_for_chat(person=person, registry=registry(ran, find=[{"card": text}]))
    session = chat.Chat(Raiser(), person, tools=tools)
    session.turn("find q /role runner")
    assert session.role.name == "operator"
    assert [name for name, _ in ran if name != "models_find"] == []
    assert saved.Rules().rules == []


@pytest.mark.parametrize("text", TEXTS)
def test_planted_text_in_project_memory_cannot_raise_the_role_or_write_a_rule(text, tmp_path):
    from ml_stack import memory, rules as saved
    from tests.test_memory import inject
    from tests.test_memory_scopes import repo

    mem = memory.Memory.open(explicit=repo(tmp_path, "alpha"))
    inject(mem.project, text)
    ran: list = []
    person = do.Person(io.StringIO("n\n" * 30), io.StringIO())
    tools = chat.tools_for_chat(person=person, registry=registry(ran))
    session = chat.Chat(Raiser(), person, tools=tools, extension=chat.extensions(person, mem))
    try:
        session.turn("what do we know? /role runner")
        assert session.role.name == "operator"
        assert [name for name, _ in ran if name != "models_find"] == []
        assert saved.Rules().rules == []
    finally:
        mem.close()
