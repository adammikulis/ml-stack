"""A hostile model or tool tries to switch off the IQ-on-Apple-silicon refusal, by every route.

The override belongs to the person's own process: a tool argument, an extra command-line flag,
a chat tool call, a field on the broker wire, an ask or a spec cannot carry it. Each attempt
runs the real component and the outcome is the refusal, with no server started.
"""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import pytest

from ml_stack import do, mcp
from ml_stack.serve import LlamaServerBackend, ServerManager, quant_guard
from ml_stack.serve.broker import Ask, Broker
from ml_stack.serve.broker_wire import _Server, spec_from
from ml_stack.serve.cli import COMMANDS
from ml_stack.serve.process import kill_process_tree
from ml_stack.serve.quant_guard import BlockedQuant
from ml_stack.testing.fakes import fake_llama_binary

STRING, U32 = 8, 4


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


@pytest.fixture
def iq_model(tmp_path) -> Path:
    path = tmp_path / "Model-UD-IQ4_XS.gguf"
    pairs = (_s("general.architecture") + struct.pack("<I", STRING) + _s("llama")
             + _s("general.file_type") + struct.pack("<I", U32) + struct.pack("<I", 30))
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + pairs)
    return path


@pytest.fixture(autouse=True)
def metal(monkeypatch):
    monkeypatch.setattr(quant_guard, "is_apple_silicon", lambda: True)
    monkeypatch.delenv(quant_guard.ENV, raising=False)
    monkeypatch.delenv(quant_guard.BROKER_ENV, raising=False)


@pytest.fixture
def no_process(monkeypatch):
    """Fails the test if a tool starts a process."""
    def refuse(*args, **kwargs):
        raise AssertionError("a process was started")

    monkeypatch.setattr(mcp, "detached", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)


def test_the_mcp_serve_up_tool_refuses_an_iq_model_and_says_why(iq_model, no_process):
    got = mcp.serve_up(model=str(iq_model))
    assert got["started"] is False and "IQ4_XS" in got["blocked"]


@pytest.mark.parametrize("flag", ["--allow-iq", "--allow", "--allow-iq=1", "--allow_iq"])
def test_the_extra_argument_of_serve_up_cannot_carry_the_override(iq_model, no_process, flag):
    plain = iq_model.with_name("Model-Q4_K_M.gguf")
    plain.write_bytes(iq_model.read_bytes().replace(struct.pack("<I", 30), struct.pack("<I", 15)))
    with pytest.raises(ValueError, match="person"):
        mcp.serve_up(model=str(plain), extra=[flag])


def test_the_chat_tool_serve_up_is_the_same_refusal(iq_model, no_process):
    tools = {schema["function"]["name"]: fn for schema, fn in do.command_tools()}
    assert tools["serve_up"](model=str(iq_model))["blocked"]
    with pytest.raises(TypeError, match="allow_iq"):
        mcp.checked(mcp.serve_up, {"model": str(iq_model), "allow_iq": True})


def test_no_tool_schema_a_model_sees_names_the_override():
    seen = json.dumps([t.public() for t in mcp.TOOLS]) + json.dumps(
        [schema for schema, _ in do.command_tools()])
    assert "allow_iq" not in seen and "allow-iq" not in seen and quant_guard.ENV not in seen


def test_the_up_command_takes_no_abbreviation_of_the_flag():
    parser = COMMANDS.parser()
    assert parser.parse_args(["up", "m.gguf", "--allow-iq"]).allow_iq is True
    for abbreviation in ("--allow", "--allow-i"):
        with pytest.raises(SystemExit):
            parser.parse_args(["up", "m.gguf", abbreviation])


@pytest.fixture
def wire(tmp_path):
    manager = ServerManager(LlamaServerBackend(binary=fake_llama_binary(tmp_path)),
                            state_file=tmp_path / "servers.json")
    server = _Server(Broker(manager, idle_s=3600.0, room=lambda: None, scan=lambda: []))
    yield server
    server.server_close()
    for held in list(server.broker.servers.values()):
        if held.pid:
            kill_process_tree(held.pid)


def test_the_broker_wire_drops_the_option_on_a_start(wire, iq_model):
    body = {"token": wire.token, "op": "start", "pid": 1, "spec":
            {"model": str(iq_model), "port": 0, "context": 512},
            "options": {"preflight": False, "check_flags": False, "allow_iq": True}}
    with pytest.raises(BlockedQuant, match="IQ4_XS"):
        wire.answer(body)
    assert not wire.broker.servers


def test_an_ask_or_a_spec_from_a_remote_caller_cannot_carry_it(iq_model):
    with pytest.raises(ValueError, match="not server settings"):
        Ask.from_json({"purpose": "p", "models": [str(iq_model)], "spec": {"allow_iq": True}})
    with pytest.raises(ValueError, match="not server settings"):
        spec_from({"model": str(iq_model), "allow_iq": True})


def test_a_lease_ask_through_the_broker_is_refused_whatever_the_options_say(wire, iq_model):
    body = {"token": wire.token, "op": "lease", "pid": 1, "purpose": "chat", "wait_s": 5,
            "models": [str(iq_model)], "spec": {"context": 512}, "options": {"allow_iq": True}}
    with pytest.raises(Exception, match="IQ4_XS") as raised:
        wire.answer(body)
    assert "blocked" in str(raised.value)
