"""The Claude Agent SDK on a served model: the same lease and environment as the launcher,
with a Python face and what each task spent."""

import contextlib
import sys
import types

import pytest

from ml_stack import harness


class _Text:
    def __init__(self, text): self.text = text


class _Assistant:
    def __init__(self, *texts): self.content = [_Text(t) for t in texts]


class _Result:
    def __init__(self):
        self.usage = {"input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 100}
        self.num_turns = 2
        self.duration_ms = 1500
        self.session_id = "s1"
        self.subtype = "success"
        self.result = "done"
        self.is_error = False


_Assistant.__name__ = "AssistantMessage"
_Result.__name__ = "ResultMessage"


@pytest.fixture
def fake_sdk(monkeypatch):
    seen = {}

    class Options:
        def __init__(self, **kw): seen["options"] = kw

    async def query(*, prompt, options=None, transport=None):
        seen["prompt"] = prompt
        yield _Assistant("Reading.")
        yield _Assistant("It is a lattice.")
        yield _Result()

    module = types.ModuleType("claude_agent_sdk")
    module.ClaudeAgentOptions = Options
    module.query = query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    return seen


def test_ask_runs_one_task_and_says_what_it_spent(fake_sdk):
    agent = harness.Harness("http://127.0.0.1:8899", "kestrel-8B", options={"max_turns": 3})
    answer = agent.ask("what is this?", allowed_tools=["Read"])
    assert answer.text == "Reading.\nIt is a lattice." and not answer.is_error
    assert answer.spent.input_tokens == 120 and answer.spent.cache_read_tokens == 100
    assert answer.spent.turns == 2 and "2 turn(s)" in answer.spent.said()
    options = fake_sdk["options"]
    assert options["model"] == "kestrel-8B" and options["max_turns"] == 3
    assert options["allowed_tools"] == ["Read"]
    assert options["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8899"
    assert options["env"]["CLAUDE_CODE_SUBAGENT_MODEL"] == "kestrel-8B"
    assert options["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert fake_sdk["prompt"] == "what is this?"


def test_session_leases_the_measured_shape_and_the_command_prints_the_answer(fake_sdk, monkeypatch, capsys):
    from ml_stack.serve.profile import record

    seen = {}

    class Server:
        base_url = "http://127.0.0.1:8899"

    @contextlib.contextmanager
    def fake_serve(model, manager=None, **lease):
        seen["lease"] = lease
        yield Server()
        seen["released"] = True

    monkeypatch.setattr("ml_stack.serve.manager.serve", fake_serve)
    monkeypatch.setattr("ml_stack.serve.profile.profile_for",
                        lambda m: record("kestrel-8B-UD-Q4_K_XL.gguf", cache_type="q8_0"))
    monkeypatch.setattr("ml_stack.graph.bench.serve.find_model", lambda m: "/m/kestrel-8B-UD-Q4_K_XL.gguf")
    monkeypatch.setattr(harness, "alias_of", lambda url, model: "kestrel-8B")
    assert harness.main(["what is this?", "--model", "kestrel", "--port", "8899",
                         "--allow", "Read", "--max-turns", "2"]) == 0
    out = capsys.readouterr().out
    assert "It is a lattice." in out and "spent: 2 turn(s)" in out
    assert seen["lease"]["port"] == 8899 and seen["lease"]["cache_type_k"] == "q8_0"
    assert seen["released"]
    assert fake_sdk["options"]["allowed_tools"] == ["Read"] and fake_sdk["options"]["max_turns"] == 2
