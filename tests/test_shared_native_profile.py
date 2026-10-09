"""Installed model selections and native workers share the maintained serving profile."""

import contextlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from poolhouse import harnessing
from poolhouse.serve import ServerManager, chat_template
from poolhouse.serve.backend import LlamaServerBackend
from poolhouse.serve.broker import Ask, Broker, Held, _shape_of
from poolhouse.serve.process import kill_process_tree
from poolhouse.serve.serving import Config, Serving
from poolhouse.testing.fakes import fake_llama_binary
from poolhouse.workspace import localloop, localmodel


def test_installed_selection_keeps_snapshot_path_instead_of_hub_alias(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"GGUF")
    snapshot = tmp_path / "Qwen.gguf"
    snapshot.symlink_to(blob)
    candidate = SimpleNamespace(path=snapshot, ref="hf:owner/repo/Qwen.gguf",
                                name="Qwen.gguf", size_bytes=4)
    pick = localmodel._line(SimpleNamespace(candidate=candidate, verdict="green", reason="fits"))
    assert pick.ref == str(snapshot)
    assert Path(pick.ref).resolve() == blob


def test_native_lease_uses_shared_harness_profile_and_template(monkeypatch, tmp_path):
    model, head, template = (tmp_path / name for name in ("Qwen.gguf", "mtp.gguf", "chat.jinja"))
    model.touch()
    config = Config(serving=Serving(model=str(model), slot_context=262144, cache_type="q8_0",
                                    draft=str(head), spec_type="draft-mtp"))
    calls = []
    monkeypatch.setattr(harnessing, "config_for", lambda found, want, say:
                        calls.append((found, want)) or config)
    monkeypatch.setattr(chat_template, "written_beside", lambda found: template)
    captured = {}

    @contextlib.contextmanager
    def lease(model, **kwargs):
        captured.update(model=model, **kwargs)
        yield SimpleNamespace(lease="owned", port=51548,
                              base_url="http://127.0.0.1:51548", adopted=True)

    monkeypatch.setattr(localloop, "serve", lease)
    monkeypatch.setattr(localloop, "check_context", lambda *args: "")
    monkeypatch.setattr(localloop, "client_on", lambda url: url)
    agent = SimpleNamespace(model=str(model), ctx=262144, name="local-qwen", size_bytes=4)
    held = localloop.lease_model(agent)
    assert calls[0][0] == str(model) and calls[0][1].ctx == 262144
    assert captured["model"] == str(model)
    spec = captured
    assert spec["draft"] == str(head) and spec["spec_type"] == "draft-mtp"
    assert spec["chat_template_file"] == str(template)
    assert spec["context"] == 262144 and spec["parallel"] == 1
    assert spec["cache_type_k"] == spec["cache_type_v"] == "q8_0"
    assert spec["cache_reuse"] == 256 and spec["warmup"] is False and "port" not in spec
    assert held.lease["shared"] is True and held.lease["port"] == 51548


@pytest.mark.parametrize("key,value", [("draft", "other.gguf"),
                                       ("chat_template_file", "other.jinja"),
                                       ("cache_type_k", "q4_0"),
                                       ("context", 524288), ("spec_draft_max", 8)])
def test_shared_server_refuses_changed_profile(key, value):
    spec = {"draft": "mtp.gguf", "chat_template_file": "chat.jinja", "cache_type_k": "q8_0",
            "context": 262144, "spec_draft_max": 4}
    held = Held(port=51548, model="Qwen.gguf", shape=_shape_of(spec))
    assert held.short_of(spec) == ""
    assert held.short_of({**spec, key: value})


def test_a_known_server_without_external_head_cannot_supply_one():
    held = Held(port=51548, model="Qwen.gguf", shape=_shape_of({"draft": None}))
    assert held.short_of({"draft": "mtp.gguf"})


@pytest.mark.parametrize("key", ["draft", "chat_template_file"])
@pytest.mark.parametrize("empty", [None, ""])
def test_explicit_empty_profile_cannot_reuse_a_configured_server(key, empty):
    held = Held(port=51548, model="Qwen.gguf", shape=_shape_of({key: "configured-file"}))
    assert held.short_of({key: empty})
    assert held.short_of({}) == ""
    unknown = Held(port=51548, model="Qwen.gguf", shape={})
    assert unknown.short_of({key: empty}) == ""
    disabled = Held(port=51548, model="Qwen.gguf", shape=_shape_of({key: empty}))
    assert disabled.short_of({key: empty}) == ""


def test_installed_aliases_share_one_native_server_with_matching_profile(tmp_path, monkeypatch):
    snapshot = tmp_path / "Qwen.gguf"
    snapshot.write_bytes(b"GGUF" + bytes(64))
    candidate = SimpleNamespace(path=snapshot, ref="hf:owner/repo/Qwen.gguf",
                                name=snapshot.name, size_bytes=snapshot.stat().st_size)
    pick = localmodel._line(SimpleNamespace(candidate=candidate, verdict="green", reason="fits"))
    manager = ServerManager(LlamaServerBackend(binary=fake_llama_binary(tmp_path)),
                            state_file=tmp_path / "servers.json")
    broker = Broker(manager, idle_s=3600, room=lambda: None, scan=lambda: [])
    manager._broker = broker
    spec = {"context": 512, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
            "cache_reuse": 256, "warmup": False}
    config = Config(serving=Serving(model=pick.ref, slot_context=512, cache_type="q8_0"))
    monkeypatch.setattr(harnessing, "config_for", lambda *args: config)
    monkeypatch.setattr(Serving, "manager", lambda self: manager)
    monkeypatch.setattr(chat_template, "written_beside", lambda found: None)
    monkeypatch.setattr(localloop, "client_on", lambda url: url)
    try:
        first = broker.lease(Ask(purpose="serve:shared-profile", models=(str(snapshot),), pid=os.getppid(),
                                 label="chat", spec=spec), timeout=10)
        second = localloop.lease_model(SimpleNamespace(model=pick.ref, ctx=512, name="coding"))
        assert second.lease["shared"] and second.lease["port"] == first.port
        assert second.lease["id"] in broker.servers[first.port].holders
        assert len(broker.servers) == 1
        second.release()
        assert first.lease in broker.servers[first.port].holders
    finally:
        for held in broker.servers.values():
            if held.pid and held.ours:
                kill_process_tree(held.pid)
