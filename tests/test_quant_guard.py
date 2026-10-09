"""IQ-family quantisations on Apple silicon: one warning by default, a refusal in strict mode,
nothing when off.

Files are real GGUF headers (metadata and a tensor table, no tensor data), servers are the
fake llama-server binary started for real, and Apple silicon is the one thing stood in for.
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
from pathlib import Path

import pytest

from poolhouse import sentinel
from poolhouse.hub.discover import ModelInfo
from poolhouse.serve import LlamaServerBackend, ServerManager, ServerSpec, quant_guard, suggest
from poolhouse.serve.broker import Ask, Broker
from poolhouse.serve.broker_wire import _Server, spec_from
from poolhouse.serve.process import kill_process_tree
from poolhouse.serve.quant_guard import BlockedQuant
from poolhouse.testing.fakes import fake_llama_binary

STRING, U32 = 8, 4
Q4_K, Q8_0, IQ4_NL, IQ4_XS = 12, 8, 20, 23
FTYPE = {"Q4_K_M": 15, "IQ4_XS": 30, "IQ2_XXS": 19, "IQ4_NL": 25}


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf(path: Path, file_type: str, tensors: tuple[tuple[str, int, int], ...] = ()) -> Path:
    """A GGUF header: architecture, file type, and ``(name, rows, ggml type)`` tensors of 256
    columns."""
    pairs = (_s("general.architecture") + struct.pack("<I", STRING) + _s("llama")
             + _s("general.file_type") + struct.pack("<I", U32) + struct.pack("<I", FTYPE[file_type]))
    table = b"".join(_s(name) + struct.pack("<I", 2) + struct.pack("<QQ", 256, rows)
                     + struct.pack("<I", kind) + struct.pack("<Q", 0)
                     for name, rows, kind in tensors)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors))
                     + struct.pack("<Q", 2) + pairs + table)
    return path


@pytest.fixture
def metal(monkeypatch):
    monkeypatch.setattr(quant_guard, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(suggest, "is_apple_silicon", lambda: True)
    monkeypatch.delenv(quant_guard.ENV, raising=False)
    monkeypatch.setattr(quant_guard, "_told", False)


@pytest.fixture
def iq_model(tmp_path):
    return gguf(tmp_path / "snap" / "UD-IQ4_XS" / "Model-UD-IQ4_XS-00001-of-00002.gguf",
                "IQ4_XS", (("blk.0.ffn.weight", 64, IQ4_XS),))


# -- detection ------------------------------------------------------------------------------

def test_strict_mode_blocks_an_iq_file_with_the_reason_both_measurements_and_the_modes(metal, iq_model):
    gguf(iq_model.parent.parent / "UD-Q4_K_XL" / "Model-UD-Q4_K_XL-00001-of-00002.gguf",
         "Q4_K_M", (("blk.0.ffn.weight", 64, Q4_K),))
    gguf(iq_model.parent.parent / "UD-Q4_K_XL" / "Model-UD-Q4_K_XL-00002-of-00002.gguf", "Q4_K_M")
    gguf(iq_model.parent.parent / "Other-Q4_K_M.gguf", "Q4_K_M")
    gguf(iq_model.parent.parent / "UD-IQ3_XXS" / "Model-UD-IQ3_XXS.gguf", "IQ4_XS")
    with pytest.raises(BlockedQuant) as raised:
        quant_guard.enforce(iq_model, asked="block")
    said = str(raised.value)
    assert "IQ4_XS" in said and "MAY be slower and less accurate" in said
    assert "70.1 s" in said and "43.7 s" in said and "ten questions" in said
    assert "27.6 s" in said and "36.6 s" in said and "pending" in said
    assert "Model-UD-Q4_K_XL-00001-of-00002.gguf" in said
    assert "00002-of-00002" not in said and "Other-Q4_K_M" not in said
    assert "IQ3_XXS" not in said
    assert "--iq warn" in said and "POOLHOUSE_IQ=off" in said


def test_iq_tensors_holding_most_of_the_bytes_block_a_file_whose_type_says_otherwise(metal, tmp_path):
    path = gguf(tmp_path / "Model-Q4_K_M.gguf", "Q4_K_M",
                (("a", 80, IQ4_XS), ("b", 20, Q4_K)))
    found = quant_guard.iq_quant(path)
    assert found is not None and found.basis == "tensors" and found.name == "IQ4_XS"
    assert 0.5 < found.share < 1


def test_a_dynamic_q4_k_xl_with_an_iq4_nl_lookup_table_is_not_blocked(metal, tmp_path):
    path = gguf(tmp_path / "Model-UD-Q4_K_XL.gguf", "Q4_K_M",
                (("per_layer_token_embd.weight", 25, IQ4_NL), ("blk.0.ffn.weight", 75, Q4_K)))
    assert quant_guard.iq_quant(path) is None
    assert quant_guard.enforce(path, asked="block") is None


def test_a_split_gguf_is_read_across_its_shards(metal, tmp_path):
    first = gguf(tmp_path / "Model-00001-of-00002.gguf", "Q4_K_M", (("small", 1, Q4_K),))
    gguf(tmp_path / "Model-00002-of-00002.gguf", "Q4_K_M", (("big", 200, IQ4_XS),))
    found = quant_guard.iq_quant(first)
    assert found is not None and found.basis == "tensors"


def test_the_file_name_decides_only_for_a_model_that_is_not_a_file(metal, tmp_path):
    named = quant_guard.iq_quant("hf:owner/repo/Model-UD-IQ4_XS-00001-of-00003.gguf")
    assert named is not None and named.basis == "file name" and named.name == "IQ4_XS"
    liar = gguf(tmp_path / "Model-IQ4_XS.gguf", "Q4_K_M", (("a", 10, Q4_K),))
    assert quant_guard.iq_quant(liar) is None


@pytest.mark.parametrize("machine", ["linux or windows", "cpu layers"])
def test_nothing_is_blocked_off_metal(monkeypatch, iq_model, machine):
    monkeypatch.delenv(quant_guard.ENV, raising=False)
    monkeypatch.setattr(quant_guard, "is_apple_silicon", lambda: machine == "cpu layers")
    layers = 0 if machine == "cpu layers" else "auto"
    assert quant_guard.enforce(iq_model, gpu_layers=layers, asked="block") is None


# -- the one place leases pass --------------------------------------------------------------

@pytest.fixture
def manager(tmp_path):
    return ServerManager(LlamaServerBackend(binary=fake_llama_binary(tmp_path)),
                         state_file=tmp_path / "servers.json")


def lease(manager, model, **options):
    return manager.lease(ServerSpec(model=model, port=_port(), context=512),
                         preflight=False, check_flags=False, **options)


def _port() -> int:
    from poolhouse.serve import free_port

    return free_port()


def test_a_strict_lease_is_refused_and_starts_nothing(metal, manager, iq_model, tmp_path, monkeypatch):
    monkeypatch.setenv(quant_guard.ENV, "block")
    with pytest.raises(BlockedQuant, match="IQ4_XS"):
        lease(manager, str(iq_model))
    assert not (tmp_path / "servers.json").exists() or json.loads(
        (tmp_path / "servers.json").read_text() or "{}") == {}


def test_the_lease_option_asks_for_strictness_too(metal, manager, iq_model):
    with pytest.raises(BlockedQuant):
        lease(manager, str(iq_model), iq="block")


def test_by_default_the_lease_proceeds_with_one_warning_and_an_event_each_time(
        metal, manager, iq_model, caplog):
    caplog.set_level(logging.WARNING, logger=quant_guard.__name__)
    infos = [lease(manager, str(iq_model)) for _ in range(2)]
    try:
        warnings = [r for r in caplog.records if "IQ quantisation" in r.getMessage()]
        assert len(warnings) == 1 and "MAY be slower" in warnings[0].getMessage()
        events = sentinel.default().bus.recent(kind="serve.iq_warning")
        assert len(events) == 2
        assert events[0].evidence["quant"] == "IQ4_XS" and events[0].evidence["who"]
        record = json.loads(manager.state_file.read_text())[str(infos[0].port)]
        assert record["iq_warning"] == "IQ4_XS"
    finally:
        for info in infos:
            kill_process_tree(info.pid)


def test_off_says_nothing_and_records_nothing(metal, manager, iq_model, caplog, monkeypatch):
    monkeypatch.setenv(quant_guard.ENV, "off")
    caplog.set_level(logging.WARNING, logger=quant_guard.__name__)
    info = lease(manager, str(iq_model))
    try:
        assert not [r for r in caplog.records if "IQ quantisation" in r.getMessage()]
        assert not sentinel.default().bus.recent(kind="serve.iq_warning")
        assert "iq_warning" not in json.loads(manager.state_file.read_text())[str(info.port)]
    finally:
        kill_process_tree(info.pid)


def test_status_shows_the_warning_while_the_server_runs(metal, tmp_path, iq_model, capsys,
                                                         monkeypatch):
    from poolhouse.serve import status_cli

    monkeypatch.setenv("POOLHOUSE_BROKER_LOCAL", "1")
    manager = ServerManager(LlamaServerBackend(binary=fake_llama_binary(tmp_path)))
    info = lease(manager, str(iq_model))
    try:
        status_cli.cmd_status(argparse.Namespace(
            every=False, json=False, port=info.port, model="", context=512, parallel=1))
        assert "WARNING  IQ4_XS is an IQ quantisation" in capsys.readouterr().out
    finally:
        kill_process_tree(info.pid)


# -- the mode is the person's, not a request's ---------------------------------------------

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


def start_body(wire, model, **options):
    return {"token": wire.token, "op": "start", "pid": 1, "label": "remote",
            "spec": {"model": str(model), "port": _port(), "context": 512}, "options":
            {"preflight": False, "check_flags": False, **options}}


def test_a_request_on_the_broker_wire_cannot_lower_a_strict_broker(metal, wire, iq_model,
                                                                   monkeypatch):
    monkeypatch.setenv(quant_guard.ENV, "block")
    for asked in ("off", "warn"):
        with pytest.raises(BlockedQuant):
            wire.answer(start_body(wire, iq_model, iq=asked))


def test_a_request_on_the_broker_wire_cannot_change_a_warning_broker(metal, wire, iq_model):
    reply = wire.answer(start_body(wire, iq_model, iq="block"))
    assert reply["ok"] and sentinel.default().bus.recent(kind="serve.iq_warning")


def test_no_wire_spec_or_ask_carries_the_mode(iq_model):
    with pytest.raises(ValueError, match="not server settings"):
        spec_from({"model": str(iq_model), "iq": "off"})
    with pytest.raises(ValueError, match="not server settings"):
        Ask.from_json({"purpose": "x", "models": [str(iq_model)], "spec": {"iq": "off"}})


# -- pickers --------------------------------------------------------------------------------

def installed(name: str, quant: str, size: int, params: int = 30_000_000_000) -> ModelInfo:
    return ModelInfo(id=f"/m/{name}", name=name, path=Path(f"/m/{name}"), format="gguf",
                     size_bytes=size, source="test", quantization=quant, parameters=params,
                     architecture="llama", context_length=8192)


def test_iq_is_only_a_tie_break_on_apple_silicon_and_is_never_dropped(metal, monkeypatch):
    from poolhouse.hub.probe import MachineMemory

    roomy = MachineMemory(total_ram=512 * 2**30, available_ram=500 * 2**30, unified=True,
                          gpu_limit_bytes=400 * 2**30)
    big_iq = installed("Model-UD-IQ4_XS", "IQ4_XS", 15 * 2**30, 60_000_000_000)
    tied_iq = installed("A-Model-IQ4_XS", "IQ4_XS", 15 * 2**30)
    tied_k = installed("Z-Model-MXFP4", "MXFP4", 15 * 2**30)
    ranked = suggest.suggest_model([tied_iq, tied_k, big_iq], roomy)
    assert [r.candidate.name for r in ranked] == [
        "Model-UD-IQ4_XS", "Z-Model-MXFP4", "A-Model-IQ4_XS"]
    assert "may be slower on Metal" in ranked[0].reason
    monkeypatch.setattr(suggest, "is_apple_silicon", lambda: False)
    off = suggest.suggest_model([tied_iq, tied_k], roomy)
    assert [r.candidate.name for r in off] == ["A-Model-IQ4_XS", "Z-Model-MXFP4"]
    monkeypatch.setattr(suggest.hub, "discover", lambda **_: [big_iq])
    monkeypatch.setattr(suggest, "is_apple_silicon", lambda: True)
    assert [r.choice.candidate.name for r in suggest.recommend(roomy)] == ["Model-UD-IQ4_XS"]


# -- internal uses --------------------------------------------------------------------------

def test_the_smoke_test_serves_an_iq_model_with_no_flag(metal, tmp_path, monkeypatch, caplog):
    from poolhouse.serve import llamacpp_smoke

    monkeypatch.setattr(llamacpp_smoke.home, "user_home", lambda: tmp_path / "user")
    root = tmp_path / "user" / ".cache" / "huggingface" / "hub"
    only = gguf(root / "small-IQ2_XXS.gguf", "IQ2_XXS", (("a", 4, 19),))
    assert llamacpp_smoke.smallest_model() == only
    got = llamacpp_smoke.run(fake_llama_binary(tmp_path), only, timeout=60)
    assert next(c for c in got.checks if c[0] == "lease")[1] is True


# -- the command ----------------------------------------------------------------------------

def test_poolhouse_serve_up_takes_the_mode_and_exits_3_when_strict(
        metal, tmp_path, iq_model, monkeypatch, capsys):
    from poolhouse.serve import lifecycle_cli, ops
    from poolhouse.serve.cli import COMMANDS

    monkeypatch.setenv("POOLHOUSE_BROKER_LOCAL", "1")
    manager = ServerManager(LlamaServerBackend(binary=fake_llama_binary(tmp_path)),
                            state_file=tmp_path / "servers.json")
    manager._broker = Broker(manager, scan=lambda: [], room=lambda: None)
    monkeypatch.setattr(ops, "manager_for", lambda *_: manager)
    single = gguf(tmp_path / "one" / "Model-UD-IQ4_XS.gguf", "IQ4_XS",
                  (("blk.0.ffn.weight", 64, IQ4_XS),))

    def run(*more):
        args = COMMANDS.parser().parse_args(
            ["up", str(single), "--port", str(_port()), "--context", "512", "--no-profile",
             "--json", *more])
        return lifecycle_cli.cmd_up(args)

    assert run("--iq", "block") == 3
    assert "blocked" in capsys.readouterr().err
    try:
        assert run() == 0
        assert sentinel.default().bus.recent(kind="serve.iq_warning")
    finally:
        for entry in json.loads(manager.state_file.read_text()).values():
            kill_process_tree(entry["pid"])


def test_a_strict_client_refuses_before_any_broker_is_asked(metal, tmp_path, iq_model):
    class Machine:
        asked = 0

        def start(self, *args, **kwargs):
            Machine.asked += 1
            raise AssertionError("the broker was asked")

    manager = ServerManager(LlamaServerBackend(binary=fake_llama_binary(tmp_path)),
                            state_file=tmp_path / "servers.json", broker=Machine())
    with pytest.raises(BlockedQuant):
        lease(manager, str(iq_model), iq="block")
    assert Machine.asked == 0
