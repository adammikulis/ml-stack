"""The ``ml-stack-models`` commands over a tree of installed models."""

from __future__ import annotations

import json

import pytest
from conftest import LLAMA_SERVER_HELP
from test_hub_discover import hf_repo, ollama

from ml_stack import hub
from ml_stack.hub.probe import GIB, MachineMemory
from ml_stack.serve import models_cli
from ml_stack.serve.backend import LlamaServerBackend, ServerSpec
from ml_stack.serve.manager import reusing_installed
from ml_stack.serve.suggest import suggest_meta
from ml_stack.testing.fakes import fake_binary


@pytest.fixture
def machine(tmp_path, monkeypatch):
    cache = tmp_path / "hf"
    hf_repo(cache, "maker/thing-GGUF", {"thing-Q4_K_M.gguf": 4000, "mmproj-thing-F16.gguf": 9})
    ollama(tmp_path / "ollama", "llama3", "latest")
    monkeypatch.setattr(hub, "default_roots", lambda _root: [cache, tmp_path / "ollama"])
    monkeypatch.setattr(hub, "machine_memory", lambda: MachineMemory(
        16 * GIB, 12 * GIB, True, 12 * GIB, (), "Darwin"))
    return tmp_path


def run(capsys, *argv):
    code = models_cli.run(list(argv))
    return code, capsys.readouterr().out


def test_list_prints_every_installed_model_and_json_is_parseable(machine, capsys):
    code, out = run(capsys, "list")
    assert code == 0 and "thing-Q4_K_M" in out and "llama3:latest" in out and "2 models" in out
    _, out = run(capsys, "list", "--json", "--source", "extra", "--format", "gguf")
    assert "llama3:latest" in [m["name"] for m in json.loads(out)]
    _, out = run(capsys, "list", "--json", "--source", "ollama")
    assert json.loads(out) == []


def test_where_lists_every_folder_with_whether_it_is_there(machine, capsys):
    _, out = run(capsys, "where", "--json")
    rows = json.loads(out)
    assert {r["path"] for r in rows} == {str(machine / "hf"), str(machine / "ollama")}
    assert all(r["present"] for r in rows)


def test_which_says_where_a_reference_would_come_from(machine, capsys):
    ref = "hf:maker/thing-GGUF/thing-Q4_K_M.gguf"
    _, out = run(capsys, "which", ref, "--json")
    got = json.loads(out)
    assert got["reused"] is True and got["path"].endswith("thing-Q4_K_M.gguf")
    _, out = run(capsys, "which", "hf:maker/other/x.gguf", "--json")
    assert json.loads(out)["download"] is True
    _, out = run(capsys, "which", "llama3:latest", "--json")
    assert json.loads(out)["reused"] is True


def test_info_gives_the_header_facts_and_an_estimate(machine, capsys):
    _, out = run(capsys, "info", "thing-Q4_K_M.gguf", "--json")
    row = json.loads(out)
    assert row["architecture"] == "llama" and row["mmproj"].endswith("mmproj-thing-F16.gguf")
    assert row["estimate_at_4k"]["context"] == 4096
    assert models_cli.run(["info", "no-such-model"]) == 1


def test_fit_prints_the_breakdown_and_a_verdict(machine, capsys):
    _, out = run(capsys, "fit", "thing-Q4_K_M.gguf", "--context", "4096", "--json")
    row = json.loads(out)
    assert row["verdict"] == "green" and row["kv_cache_type"] == "q8_0"
    assert [b["label"] for b in row["breakdown"]][:2] == ["Weights", "KV cache"]
    assert row["meters"][0]["capacity_bytes"] > 0 and row["meters"][0]["segments"]
    assert row["max_context"] == 4096
    _, text = run(capsys, "fit", "thing-Q4_K_M.gguf")
    assert "verdict green" in text and "Weights" in text


def test_suggest_prints_settings_and_reasons(machine, capsys):
    _, out = run(capsys, "suggest", "thing-Q4_K_M.gguf", "--goal", "chat")
    assert "kv q8_0" in out and "Context" in out and "alternative" in out


def test_recommend_ranks_what_is_installed(machine, capsys):
    _, out = run(capsys, "recommend", "--json")
    rows = json.loads(out)
    assert rows and all(r["installed"] for r in rows)


def test_the_command_a_suggestion_leases_asks_for_a_quantised_cache(tmp_path):
    found = {"general.architecture": "llama", "llama.block_count": 32,
             "llama.context_length": 32768, "llama.attention.head_count": 32,
             "llama.attention.head_count_kv": 8, "llama.attention.key_length": 128,
             "llama.attention.value_length": 128, "llama.embedding_length": 4096}
    big = MachineMemory(64 * GIB, 50 * GIB, True, 48 * GIB, (), "Darwin")
    lease = suggest_meta(found, 4 * GIB, big).lease()
    spec = ServerSpec(model="m.gguf", port=8123, **lease)
    argv = LlamaServerBackend(binary=fake_binary(tmp_path, help_text=LLAMA_SERVER_HELP)
                              ).command(spec)
    assert argv[argv.index("--cache-type-k") + 1] == "q8_0"
    assert argv[argv.index("--cache-type-v") + 1] == "q8_0"
    assert argv[argv.index("-fa") + 1] == "on" and argv[argv.index("-c") + 1] == "32768"
    assert argv[argv.index("-ub") + 1] == "2048"


def test_without_flash_attention_only_k_is_quantised_in_the_command():
    found = {"general.architecture": "llama", "llama.block_count": 8,
             "llama.context_length": 8192, "llama.attention.head_count": 32,
             "llama.attention.head_count_kv": 8, "llama.attention.key_length": 72,
             "llama.attention.value_length": 72, "llama.embedding_length": 2304}
    big = MachineMemory(64 * GIB, 50 * GIB, True, 48 * GIB, (), "Darwin")
    lease = suggest_meta(found, GIB, big).lease()
    assert (lease["cache_type_k"], lease["cache_type_v"], lease["flash_attn"]) == (
        "q8_0", "f16", False)


def test_serving_reuses_an_installed_model_instead_of_downloading(machine):
    spec = ServerSpec(model="hf:maker/thing-GGUF/thing-Q4_K_M.gguf", port=8123,
                      mmproj="hf:maker/thing-GGUF/mmproj-thing-F16.gguf")
    got = reusing_installed(spec)
    assert got.model.endswith("thing-Q4_K_M.gguf") and not str(got.model).startswith("hf:")
    assert str(got.mmproj).endswith("mmproj-thing-F16.gguf")
    other = ServerSpec(model="hf:maker/thing-GGUF/thing-Q8_0.gguf", port=8123)
    assert reusing_installed(other) is other


def test_a_name_ollama_gave_a_model_finds_its_blob(machine):
    found = hub.located("llama3:latest")
    assert found is not None and found.name.startswith("sha256-")
    assert hub.located("llama3") is None
    assert hub.located("llama3", loose=True) == found


def test_the_ollama_blob_is_served_by_path(machine, tmp_path):
    path = hub.located("llama3:latest")
    backend = LlamaServerBackend(binary=fake_binary(tmp_path, help_text=LLAMA_SERVER_HELP))
    argv = backend.command(ServerSpec(model=str(path), port=8123))
    assert argv[argv.index("-m") + 1] == str(path)
