"""Models live in the Hugging Face hub cache: a pull writes its layout, and serving finds it there."""

from __future__ import annotations

import hashlib
import struct

import pytest

from poolhouse import home, http, hub, net
from poolhouse.hub import hfstore, transfer
from poolhouse.serve import backend
from poolhouse.testing.fakehub import fake_hub

BIG = b"GGUF" + struct.pack("<IQQ", 3, 0, 0) + bytes(range(251)) * 4096
SMALL = b"{\"architectures\": [\"x\"]}"
_HEAD = b'{"__metadata__":{}}'
TENSORS = struct.pack("<Q", len(_HEAD)) + _HEAD + bytes(64)
REPO = "maker/thing-GGUF"
REPOS = {REPO: {"thing-Q4_K_M.gguf": BIG, "README.md": SMALL},
         "maker/st": {"config.json": SMALL, "model.safetensors": TENSORS, "weights.bin": b"pickle"}}


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(http, "check", lambda url: url)
    with fake_hub(REPOS) as hub_:
        hub_.plain.add(f"{REPO}/README.md")
        hub_.point(monkeypatch)
        yield hub_


def snapshot_dir(server, tmp_path, repo=REPO):
    return tmp_path / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / server.commit(repo)


def test_a_pull_lands_in_the_hub_cache_layout(server, tmp_path):
    got = hub.pull(f"hf:{REPO}/thing-Q4_K_M.gguf")
    root = tmp_path / "hub" / "models--maker--thing-GGUF"
    blob = root / "blobs" / hashlib.sha256(BIG).hexdigest()
    assert got == snapshot_dir(server, tmp_path) / "thing-Q4_K_M.gguf"
    assert got.is_symlink() and got.readlink().as_posix() == f"../../blobs/{blob.name}"
    assert blob.read_bytes() == BIG and got.read_bytes() == BIG
    assert (root / "refs" / "main").read_text() == server.commit(REPO)
    assert not list((tmp_path / "hub").rglob("*.part"))
    assert not home.state("models").exists()


def test_a_small_file_is_named_by_its_git_object_id_and_checked_against_it(server, tmp_path):
    got = hub.pull(f"hf:{REPO}/README.md")
    oid = hashlib.sha1(b"blob %d\0" % len(SMALL) + SMALL, usedforsecurity=False).hexdigest()
    assert (tmp_path / "hub" / "models--maker--thing-GGUF" / "blobs" / oid).read_bytes() == SMALL
    assert got.read_bytes() == SMALL


def test_a_file_that_differs_from_its_object_id_is_refused(server, tmp_path):
    server.corrupt.add(f"{REPO}/README.md")
    with pytest.raises(net.Blocked, match="object id"):
        hub.pull(f"hf:{REPO}/README.md")
    assert not list((tmp_path / "hub").rglob("README.md"))
    assert not list((tmp_path / "hub").glob("models--*/blobs/*"))


def test_a_wrong_sha256_leaves_nothing_in_the_cache(server, tmp_path):
    server.corrupt.add(f"{REPO}/thing-Q4_K_M.gguf")
    with pytest.raises(transfer.ChecksumMismatch):
        hub.pull(f"hf:{REPO}/thing-Q4_K_M.gguf")
    assert not list((tmp_path / "hub").glob("models--*/blobs/*"))
    assert not list((tmp_path / "hub").glob("models--*/snapshots/*/*"))


def test_a_second_pull_downloads_nothing(server):
    hub.pull(f"hf:{REPO}/thing-Q4_K_M.gguf")
    before = dict(server.downloads)
    hub.pull(f"hf:{REPO}/thing-Q4_K_M.gguf")
    assert server.downloads == before


def test_the_cache_folder_is_the_one_hf_names(server, tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    got = hub.pull(f"hf:{REPO}/thing-Q4_K_M.gguf")
    assert got.is_relative_to(tmp_path / "hfhome" / "hub")
    monkeypatch.delenv("HF_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert hub.pull(f"hf:{REPO}/thing-Q4_K_M.gguf").is_relative_to(tmp_path / "xdg" / "huggingface" / "hub")


def test_serving_resolves_a_reference_through_the_cache(server, tmp_path):
    path = backend.fetched(f"hf:{REPO}/thing-Q4_K_M.gguf", "model")
    assert path == str(snapshot_dir(server, tmp_path) / "thing-Q4_K_M.gguf")
    assert hub.located("thing-Q4_K_M.gguf") == snapshot_dir(server, tmp_path) / "thing-Q4_K_M.gguf"
    assert hub.repo_of(path) == REPO


def test_fetch_returns_the_snapshot_path(server, tmp_path):
    assert hub.fetch(f"hf:{REPO}/thing-Q4_K_M.gguf") == snapshot_dir(server, tmp_path) / "thing-Q4_K_M.gguf"


def test_a_snapshot_is_the_cache_folder_without_pickles_and_is_held_afterwards(server, tmp_path):
    folder = hub.snapshot("maker/st")
    assert folder == snapshot_dir(server, tmp_path, "maker/st")
    assert sorted(p.name for p in folder.iterdir()) == ["config.json", "model.safetensors"]
    assert hub.held_snapshot("maker/st") == folder
    assert hub.held_snapshot("maker/none") is None


def test_a_planted_ref_is_not_followed_out_of_the_cache(server, tmp_path):
    first = hub.snapshot("maker/st")
    refs = first.parent.parent / "refs" / "main"
    for planted in ("../../../../..", "..", "", "x" * 40, first.name.upper()):
        refs.write_text(planted)
        assert hfstore.main_commit("maker/st") == ""
        assert hub.held_snapshot("maker/st") == first        # the newest snapshot, never the planted path
    refs.write_text(first.name + "\n")
    assert hfstore.main_commit("maker/st") == first.name


def test_an_existing_snapshot_symlink_is_replaced_not_followed(tmp_path):
    blob = tmp_path / "blobs" / "a"
    blob.parent.mkdir()
    blob.write_bytes(b"x")
    target = tmp_path / "snap" / "f"
    target.parent.mkdir()
    target.symlink_to("wrong")
    hfstore.link(target, blob)
    assert target.read_bytes() == b"x" and target.readlink().as_posix() == "../blobs/a"
