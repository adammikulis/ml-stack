"""Public Hub snapshot bounds, digests and cache destinations."""

import hashlib
from types import SimpleNamespace

import pytest
from test_hub_discover import symlink

from ml_stack.hub import hf_cache

COMMIT = "a" * 40


def metadata(name, data, *, lfs=False):
    row = {"rfilename": name, "size": len(data)}
    if lfs:
        row["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    else:
        row["blobId"] = hashlib.sha1(f"blob {len(data)}\0".encode() + data,
                                      usedforsecurity=False).hexdigest()
    return row


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    data = {"sha": COMMIT, "siblings": [metadata("README.md", b"card"),
            metadata("data/train.parquet", b"data", lfs=True),
            {"rfilename": "pytorch_model.bin"}]}
    calls = []
    monkeypatch.setattr(hf_cache.hub, "hub_cache", lambda: cache)
    monkeypatch.setattr(hf_cache.remote, "endpoint", lambda: "https://hub.example")
    monkeypatch.setattr(hf_cache.remote, "token", lambda: pytest.fail("private token read"))

    def info(url, ask):
        assert ask.token == "" and ask.max_bytes == 8 << 20
        calls.append(url)
        return data

    def download(url, target, want):
        assert want.token == "" and want.admit
        payload = b"card" if target.name == "README.md" else b"data"
        target.write_bytes(payload)
        assert want.verify(target, {}) == ""
        assert want.max_bytes == len(payload)
        calls.append(url)

    monkeypatch.setattr(hf_cache.net, "default", lambda: SimpleNamespace(json=info))
    monkeypatch.setattr(hf_cache.net, "download", download)
    return cache, data, calls


def test_dataset_snapshot_is_revisioned_public_and_reuses_verified_files(snapshot, monkeypatch):
    cache, _, calls = snapshot
    monkeypatch.setenv("HF_TOKEN", "private-test-token")
    folder = hf_cache.fetch("sample/dataset", repo_type="dataset")
    assert folder == cache / "datasets--sample--dataset" / "snapshots" / COMMIT
    assert (folder / "data/train.parquet").read_bytes() == b"data"
    assert not (folder / "pytorch_model.bin").exists()
    assert (folder.parents[1] / "refs/main").read_text() == COMMIT
    assert calls == ["https://hub.example/api/datasets/sample/dataset/revision/main?blobs=true",
                     f"https://hub.example/datasets/sample/dataset/resolve/{COMMIT}/README.md",
                     f"https://hub.example/datasets/sample/dataset/resolve/{COMMIT}/data/train.parquet"]
    hf_cache.fetch("sample/dataset", repo_type="dataset")
    assert len(calls) == 4
    (folder / "README.md").write_bytes(b"evil")
    hf_cache.fetch("sample/dataset", repo_type="dataset")
    assert (folder / "README.md").read_bytes() == b"card" and len(calls) == 6


@pytest.mark.parametrize("name", ["../escape", "/escape", "C:/escape", "data\\escape",
                                 "data//escape", "NUL.txt", "data/.", "data/trailing."])
def test_remote_file_names_cannot_escape_or_alias(snapshot, name):
    cache, data, _ = snapshot
    data["siblings"] = [metadata(name, b"card")]
    with pytest.raises(ValueError):
        hf_cache.fetch("sample/model")
    assert not cache.exists()


@pytest.mark.parametrize("change", [{"sha": "../elsewhere"}, {"sha": "b" * 40},
                                    {"siblings": "files"}, {"siblings": []}])
def test_malformed_or_wrong_pinned_revision_is_refused(snapshot, change):
    cache, data, _ = snapshot
    data.update(change)
    with pytest.raises(ValueError):
        hf_cache.fetch("sample/model", revision=COMMIT)
    assert not cache.exists()


def test_all_file_metadata_is_validated_before_downloading(snapshot):
    cache, data, calls = snapshot
    data["siblings"].append({"rfilename": "last.json", "size": -1, "blobId": "b" * 40})
    with pytest.raises(ValueError):
        hf_cache.fetch("sample/model")
    assert len(calls) == 1 and not cache.exists()


def test_digest_verification_rejects_wrong_bytes(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"evil")
    for lfs in (False, True):
        row = metadata("file", b"card", lfs=lfs)
        digest = row["lfs"]["sha256"] if lfs else row["blobId"]
        member = hf_cache.Member("file", 4, digest, lfs)
        assert "differs" in member.verify(path, {})


def test_cache_inside_checkout_is_refused_before_network(snapshot, monkeypatch):
    cache, _, calls = snapshot
    monkeypatch.setattr(hf_cache.worktreerules, "checkouts", lambda path: (cache, cache))
    with pytest.raises(ValueError, match="outside a Git checkout"):
        hf_cache.fetch("sample/model")
    assert not cache.exists() and not calls


def test_snapshot_subdirectory_inside_nested_checkout_is_refused(snapshot, monkeypatch):
    cache, _, calls = snapshot
    monkeypatch.setattr(hf_cache.worktreerules, "checkouts",
                        lambda path: None if path == cache else (path, path))
    with pytest.raises(ValueError, match="outside a Git checkout"):
        hf_cache.fetch("sample/model")
    assert not cache.exists() and len(calls) == 1


def test_cached_snapshot_symlink_cannot_leave_repository(snapshot, tmp_path):
    cache, _, calls = snapshot
    root = cache / "models--sample--model"
    folder = root / "snapshots" / COMMIT
    folder.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"card")
    symlink(folder / "README.md", outside)
    with pytest.raises(ValueError):
        hf_cache.fetch("sample/model")
    assert len(calls) == 1 and outside.read_bytes() == b"card"


def test_member_nested_checkout_refused_before_any_download(snapshot, monkeypatch):
    cache, _, calls = snapshot
    monkeypatch.setattr(hf_cache.worktreerules, "checkouts",
                        lambda path: (path, path) if path.name == "train.parquet" else None)
    with pytest.raises(ValueError, match="outside a Git checkout"):
        hf_cache.fetch("sample/model")
    assert len(calls) == 1 and not cache.exists()
