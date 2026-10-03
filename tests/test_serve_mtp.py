"""Multi-token prediction is on by default: a model that carries its own prediction layer, or
ships a trusted and compatible head, is started with `--spec-type draft-mtp` and everything
else is served as before, with the reason said."""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import gguf
import numpy as np
import pytest

from ml_stack import sentinel
from ml_stack.serve import LlamaServerBackend, ServerManager, ServerSpec, free_port, mtp
from ml_stack.serve.process import every_server, pid_exists
from ml_stack.testing.fakes import LLAMA_SERVER_FLAGS, fake_llama_binary

MTP_HELP = "".join(
    f"{('--spec-type none,draft-simple,draft-mtp,ngram-mod' if flag == '--spec-type TYPE' else flag):<52}"
    "what it sets\n" for flag in LLAMA_SERVER_FLAGS)
OLD_HELP = "".join(
    f"{('--spec-type none,draft-simple,ngram-mod' if flag == '--spec-type TYPE' else flag):<52}"
    "what it sets\n" for flag in LLAMA_SERVER_FLAGS)


def a_gguf(path: Path, *, arch: str = "qwen35", width: int = 8, layer: bool = False,
           tokens: tuple[str, ...] = ("a", "b", "c", "d")) -> Path:
    """A real, tiny GGUF; ``layer`` gives it the tensors and key of a prediction layer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(path), arch)
    writer.add_embedding_length(width)
    writer.add_block_count(4)
    writer.add_uint32(f"{arch}.nextn_predict_layers", 1 if layer else 0)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen35")
    writer.add_token_list(list(tokens))
    writer.add_tensor("token_embd.weight", np.zeros((width, 4), dtype=np.float32))
    if layer:
        writer.add_tensor("blk.3.nextn.eh_proj.weight", np.zeros((2, 2), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


@pytest.fixture
def cache(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    monkeypatch.delenv(mtp.ENV, raising=False)
    return root


def snapshot(cache: Path, repo: str = "owner--model") -> Path:
    return cache / f"models--{repo}" / "snapshots" / "rev1"


def server(tmp_path: Path, help_text: str = MTP_HELP) -> tuple[Path, ServerManager]:
    where = tmp_path / "bin"
    where.mkdir(exist_ok=True)
    binary = fake_llama_binary(where, help_text=help_text)
    manager = ServerManager(LlamaServerBackend(binary=binary), state_file=tmp_path / "leases.json")
    return binary, manager


def leased(manager: ServerManager, model: Path, **over):
    spec = ServerSpec(model=model, port=free_port(), context=512, **over)
    return manager.lease(spec, timeout=60, preflight=False, warmup_request=False)


def argv_of(binary: Path) -> list[str]:
    return json.loads((binary.parent / "argv.json").read_text())


def after(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


@pytest.fixture
def stop(tmp_path):
    held: list[tuple[ServerManager, object]] = []
    yield lambda manager, info: held.append((manager, info))
    for manager, info in held:
        manager.release(info)
        deadline = time.monotonic() + 10
        while info.pid and pid_exists(info.pid) and time.monotonic() < deadline:
            time.sleep(0.1)


def test_the_weights_own_prediction_layer_is_served_without_being_asked(cache, tmp_path, stop):
    model = a_gguf(snapshot(cache) / "Own-Q4_K_M.gguf", layer=True)
    binary, manager = server(tmp_path)
    info = leased(manager, model)
    stop(manager, info)
    argv = argv_of(binary)
    assert after(argv, "--spec-type") == "draft-mtp"
    assert "-md" not in argv and "--model-draft" not in argv
    assert info.mtp == "embedded" and "MTP on" in info.mtp_note
    recorded = json.loads((tmp_path / "leases.json").read_text())[str(info.port)]
    assert recorded["mtp"] == "embedded"


def test_a_head_shipped_with_the_weights_is_served_as_the_draft(cache, tmp_path, stop):
    model = a_gguf(snapshot(cache) / "Plain-Q4_K_M.gguf")
    head = a_gguf(snapshot(cache) / "MTP" / "mtp-Plain-Q8_0.gguf", layer=True)
    binary, manager = server(tmp_path)
    info = leased(manager, model)
    stop(manager, info)
    argv = argv_of(binary)
    assert after(argv, "-md") == str(head)
    assert after(argv, "--spec-type") == "draft-mtp"
    assert info.mtp == head.name


def test_ml_stack_mtp_off_serves_without(cache, tmp_path, stop, monkeypatch):
    model = a_gguf(snapshot(cache) / "Own-Q4_K_M.gguf", layer=True)
    binary, manager = server(tmp_path)
    monkeypatch.setenv(mtp.ENV, "off")
    info = leased(manager, model)
    stop(manager, info)
    argv = argv_of(binary)
    assert "--spec-type" not in argv and "-md" not in argv
    assert info.mtp == "" and mtp.ENV in info.mtp_note


def test_a_lease_can_ask_for_none(cache, tmp_path, stop):
    model = a_gguf(snapshot(cache) / "Own-Q4_K_M.gguf", layer=True)
    binary, manager = server(tmp_path)
    info = leased(manager, model, mtp=False)
    stop(manager, info)
    assert "--spec-type" not in argv_of(binary)
    assert info.mtp == "" and "asked for none" in info.mtp_note


def test_a_build_without_draft_mtp_serves_plainly_and_says_so(cache, tmp_path, stop):
    model = a_gguf(snapshot(cache) / "Own-Q4_K_M.gguf", layer=True)
    binary, manager = server(tmp_path, OLD_HELP)
    info = leased(manager, model)
    stop(manager, info)
    assert "--spec-type" not in argv_of(binary)
    assert "no --spec-type draft-mtp" in info.mtp_note


def test_a_speculation_the_caller_chose_is_left_alone(cache, tmp_path, stop):
    model = a_gguf(snapshot(cache) / "Own-Q4_K_M.gguf", layer=True)
    binary, manager = server(tmp_path)
    info = leased(manager, model, spec_type="ngram-mod")
    stop(manager, info)
    assert after(argv_of(binary), "--spec-type") == "ngram-mod"
    assert info.mtp == ""


@pytest.mark.parametrize("over", [{"slot_save_path": "slots"}, {"embedding": True},
                                  {"spec_tree": 2}])
def test_modes_that_cannot_draft_are_served_without(cache, tmp_path, stop, over):
    model = a_gguf(snapshot(cache) / "Own-Q4_K_M.gguf", layer=True)
    binary, manager = server(tmp_path, MTP_HELP + "--spec-tree N  tree\n")
    if "slot_save_path" in over:
        over = {"slot_save_path": str(tmp_path / "slots")}
    info = leased(manager, model, **over)
    stop(manager, info)
    assert after(argv_of(binary), "--spec-type") is None
    assert info.mtp_note.startswith("MTP off")


def test_a_head_from_another_repository_is_not_trusted(cache, tmp_path, stop):
    model = a_gguf(snapshot(cache) / "Plain-Q4_K_M.gguf")
    stranger = a_gguf(snapshot(cache, "someone--else") / "MTP" / "mtp-Plain-Q8_0.gguf", layer=True)
    binary, manager = server(tmp_path)
    info = leased(manager, model)
    stop(manager, info)
    assert "-md" not in argv_of(binary) and "--spec-type" not in argv_of(binary)
    assert "different repository" in info.mtp_note and stranger.name in info.mtp_note


def test_a_head_from_another_repository_that_a_download_pinned_is_used(cache, tmp_path, stop):
    model = a_gguf(snapshot(cache) / "Plain-Q4_K_M.gguf")
    stranger = a_gguf(snapshot(cache, "someone--else") / "MTP" / "mtp-Plain-Q8_0.gguf", layer=True)
    sentinel.default().manifest.pin(stranger, "model", source="hf:someone/else")
    binary, manager = server(tmp_path)
    info = leased(manager, model)
    stop(manager, info)
    assert after(argv_of(binary), "-md") == str(stranger)


def test_a_head_for_another_model_is_refused_by_its_header(cache, tmp_path, stop):
    model = a_gguf(snapshot(cache) / "Plain-Q4_K_M.gguf")
    a_gguf(snapshot(cache) / "MTP" / "mtp-Plain-Q8_0.gguf", layer=True, width=16)
    binary, manager = server(tmp_path)
    info = leased(manager, model)
    stop(manager, info)
    assert "-md" not in argv_of(binary)
    assert "embedding_length differs" in info.mtp_note


def test_a_head_whose_bytes_no_longer_match_its_pin_is_left_out(cache, tmp_path, stop):
    model = a_gguf(snapshot(cache) / "Plain-Q4_K_M.gguf")
    head = a_gguf(snapshot(cache) / "MTP" / "mtp-Plain-Q8_0.gguf", layer=True)
    sentinel.default().manifest.pin(head, "model", source="hf:owner/model")
    data = bytearray(head.read_bytes())
    data[-1] ^= 0xFF
    head.write_bytes(bytes(data))
    binary, manager = server(tmp_path)
    info = leased(manager, model)
    stop(manager, info)
    assert "-md" not in argv_of(binary)
    assert "does not match its pin" in info.mtp_note


def test_a_server_that_will_not_start_with_the_head_is_started_without(cache, tmp_path, stop):
    model = a_gguf(snapshot(cache) / "Own-Q4_K_M.gguf", layer=True)
    (tmp_path / "inner").mkdir()
    real = fake_llama_binary(tmp_path / "inner", help_text=MTP_HELP)
    wrapper = tmp_path / "bin" / "llama-server"
    wrapper.parent.mkdir()
    wrapper.write_text(
        "#!/bin/sh\n"
        f"if [ \"$1\" = --help ]; then cat {real.parent / 'help.txt'}; exit 0; fi\n"
        "case \"$*\" in *draft-mtp*) echo 'unsupported architecture for mtp' >&2; exit 1;; esac\n"
        f"exec {real} \"$@\"\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    manager = ServerManager(LlamaServerBackend(binary=wrapper), state_file=tmp_path / "leases.json")
    info = leased(manager, model)
    stop(manager, info)
    assert "--spec-type" not in argv_of(real)
    assert info.mtp == "" and "would not start with it" in info.mtp_note
    again = mtp.plan(ServerSpec(model=model), binary=wrapper)
    assert not again.active and "failed to start" in again.note


def test_headers_are_compared_before_a_head_is_trusted(tmp_path):
    model = a_gguf(tmp_path / "m.gguf")
    assert mtp.mismatch(model, a_gguf(tmp_path / "ok.gguf", layer=True)) == ""
    assert "architecture" in mtp.mismatch(model, a_gguf(tmp_path / "a.gguf", arch="llama",
                                                         layer=True))
    assert "vocabulary" in mtp.mismatch(model, a_gguf(tmp_path / "v.gguf", layer=True,
                                                       tokens=("a", "b")))
    assert "no prediction layer" in mtp.mismatch(model, a_gguf(tmp_path / "n.gguf"))


def test_the_off_words_for_the_environment(monkeypatch):
    for word in ("off", "0", "no", "false", "OFF"):
        assert not mtp.enabled({mtp.ENV: word})
    for word in ("", "on", "1", "auto"):
        assert mtp.enabled({mtp.ENV: word})


def test_the_judge_lease_asks_for_no_mtp_and_a_none_arm_serves_without():
    from ml_stack.bench.serve import drafted_by
    from ml_stack.serve.serving import Config, Serving

    config = Config(serving=Serving(model="m.gguf"))
    assert drafted_by(config, "").lease()["mtp"] is False
    assert "mtp" not in drafted_by(config, "head.gguf").lease()
    source = Path(__file__).resolve().parent.parent / "src/ml_stack/guard/native.py"
    assert '"mtp": False' in source.read_text()


@pytest.mark.slow
def test_a_real_server_drafts_by_default_and_keeps_greedy_output(cache, tmp_path, _real_home):
    """The managed build, a small model that carries its own prediction layer, and nothing else
    running: the default serves draft-mtp, and greedy output equals that of the same model
    served with MTP off."""
    from ml_stack.client import Client

    account = _real_home.state.parent
    builds = sorted((account / ".ml-stack" / "llama.cpp" / "builds").glob("*/llama-server"))
    hub = account / ".cache" / "huggingface" / "hub"
    models = [p for p in hub.rglob("*.gguf") if mtp.embeds_head(p) and "mtp" not in p.name.lower()
              and 0 < p.stat().st_size < 4 * 1024 ** 3] if hub.is_dir() else []
    if not builds or not models:
        pytest.skip("no managed llama-server or small model with a prediction layer")
    if every_server():
        pytest.skip("another llama-server is running; a test does not load a model beside it")
    model, binary = min(models, key=lambda p: p.stat().st_size), builds[-1]
    if not mtp.supported(binary):
        pytest.skip("the managed build has no draft-mtp")
    manager = ServerManager(LlamaServerBackend(binary=binary), state_file=tmp_path / "leases.json")
    ask = [{"role": "user", "content": "Count from one to twelve in words."}]
    answers = {}
    for label, over in (("mtp", {}), ("plain", {"mtp": False})):
        info = manager.lease(ServerSpec(model=model, port=free_port(), context=2048, **over),
                             timeout=600)
        try:
            assert (info.mtp == "embedded") == (label == "mtp")
            answers[label] = Client(info.base_url).chat(ask, max_tokens=64, temperature=0).content
        finally:
            manager.release(info)
    assert answers["mtp"] == answers["plain"] != ""
    assert os.environ.get(mtp.ENV) is None
