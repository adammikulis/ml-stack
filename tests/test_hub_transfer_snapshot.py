"""Snapshot selection for Hub pulls."""

import pytest

from poolhouse.hub import transfer
from poolhouse.hub.remote import NotFound, RemoteFile


@pytest.fixture
def listed(monkeypatch):
    files = [RemoteFile(name, 1) for name in (
        "config.json", "tokenizer.json", "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors", "weights.bin", ".gitattributes")]
    monkeypatch.setattr(transfer.remote, "listing", lambda *args: files)
    calls = []
    monkeypatch.setattr(transfer, "_bring", lambda *args: calls.append(args))
    return files, calls


@pytest.mark.parametrize("ref", ["hf:maker/model", "hf:maker/model/model-00001-of-00002.safetensors"])
def test_safetensors_pull_downloads_snapshot(ref, listed, tmp_path):
    files, calls = listed
    def callback(progress):
        pass
    cancel = transfer.CancelToken()
    assert transfer.pull(ref, tmp_path, callback, cancel, peers=False) == tmp_path
    parsed, chosen, folder, run = calls[0]
    assert parsed.repo == "maker/model"
    assert chosen == files[:4]
    assert folder == tmp_path
    assert run.report is callback and run.cancel is cancel and run.peers is False


@pytest.mark.parametrize("ref", ["hf:maker/model/missing.safetensors", "hf:maker/model:Q4_K_M", "hf:maker/model/missing.gguf"])
def test_missing_explicit_reference_is_not_replaced_by_snapshot(ref, listed, tmp_path):
    with pytest.raises(NotFound):
        transfer.pull(ref, tmp_path)
    assert listed[1] == []


def test_snapshot_keeps_safe_files_without_model_configuration(listed, tmp_path):
    files, calls = listed
    files[:] = [RemoteFile("tokenizer.json", 1), RemoteFile("weights.bin", 1)]
    assert transfer.snapshot("maker/model", revision="release", dest=tmp_path) == tmp_path
    assert calls[0][0].revision == "release"
    assert calls[0][1] == files[:1]
    assert calls[0][2] == tmp_path


def test_repo_pull_requires_model_configuration(listed, tmp_path):
    files, calls = listed
    files[:] = [RemoteFile("model.safetensors", 1)]
    with pytest.raises(NotFound):
        transfer.pull("hf:maker/model", tmp_path)
    assert calls == []


@pytest.mark.parametrize("name", ["../escaped.json", "/escaped.json", "linked/escaped.json"])
def test_download_rejects_paths_outside_destination(name, tmp_path, monkeypatch):
    folder = tmp_path / "model"
    folder.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (folder / "linked").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(transfer, "_space", lambda *args: pytest.fail("disk checked before validation"))
    with pytest.raises(ValueError, match="Unsafe model file path"):
        transfer._bring(transfer.remote.Ref("maker/model", "", "", "main"),
                        [RemoteFile(name, 1)], folder, transfer._Run())
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("error", [transfer.GatedRepo, transfer.remote.RemoteError])
def test_snapshot_pull_preserves_endpoint_errors(error, monkeypatch, tmp_path):
    def refuse(*args):
        raise error("endpoint refused")
    monkeypatch.setattr(transfer.remote, "listing", refuse)
    with pytest.raises(error, match="endpoint refused"):
        transfer.pull("hf:maker/model/model.safetensors", tmp_path)


def test_gguf_pull_returns_named_file(listed, tmp_path):
    files, calls = listed
    files[:] = [RemoteFile("model-Q4_K_M.gguf", 1), RemoteFile("config.json", 1)]
    assert transfer.pull("hf:maker/model/model-Q4_K_M.gguf", tmp_path) == tmp_path / files[0].path
    assert calls[0][1] == files[:1]
