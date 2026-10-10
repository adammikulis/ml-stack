"""``poolhouse-models migrate``: the old ``<state>/models`` store moves into the Hugging Face hub cache."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import struct

import pytest

from poolhouse import files, home, http, hub
from poolhouse.serve import models_cli, models_migrate as mig
from poolhouse.testing.fakehub import fake_hub

BIG = b"GGUF" + struct.pack("<IQQ", 3, 0, 0) + bytes(range(251)) * 2048
SMALL = b'{"architectures": ["x"]}'
REPO = "maker/thing-GGUF"
GONE = "maker/vanished"
REPOS = {REPO: {"thing-Q4_K_M.gguf": BIG, "README.md": SMALL}}


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


def old_file(repo: str, name: str, body: bytes):
    path = home.state("models", *repo.split("/")) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


@pytest.fixture
def old(server):
    return {"big": old_file(REPO, "thing-Q4_K_M.gguf", BIG), "small": old_file(REPO, "README.md", SMALL)}


def run(**kw):
    return mig.migrate(apply=True, models=lambda: [], **kw)


def cache_root(tmp_path):
    return tmp_path / "hub" / "models--maker--thing-GGUF"


def test_plan_counts_and_sizes_without_hashing_or_asking_the_hub(old, tmp_path, monkeypatch, server):
    monkeypatch.setattr(files, "sha256_file", lambda *a, **k: pytest.fail("a plan hashes nothing"))
    server.hub_seen.clear()
    report = mig.migrate(apply=False, models=lambda: [])
    assert (report.mode, report.files, report.bytes, report.repos) == ("plan", 2, len(BIG) + len(SMALL), 1)
    assert server.hub_seen == [] and report.ok
    assert all(p.exists() for p in old.values()) and not (tmp_path / "hub" / "blobs").exists()
    assert any("same volume" in note for note in report.notes)


def test_plan_of_an_empty_store_says_so(server):
    report = mig.migrate(apply=False, models=lambda: [])
    assert report.files == 0 and report.ok and "nothing to move" in report.notes[0]


def test_run_moves_an_lfs_file_and_a_small_file_into_the_hub_layout(old, server, tmp_path):
    report = run()
    root, commit = cache_root(tmp_path), server.commit(REPO)
    assert report.ok and (report.moved, report.deduped) == (2, 0)
    big_blob = root / "blobs" / hashlib.sha256(BIG).hexdigest()
    small_blob = root / "blobs" / hashlib.sha1(b"blob %d\0" % len(SMALL) + SMALL, usedforsecurity=False).hexdigest()
    assert big_blob.read_bytes() == BIG and small_blob.read_bytes() == SMALL
    link = root / "snapshots" / commit / "thing-Q4_K_M.gguf"
    assert link.is_symlink() and link.readlink().as_posix() == f"../../blobs/{big_blob.name}"
    assert (root / "snapshots" / commit / "README.md").read_bytes() == SMALL
    assert (root / "refs" / "main").read_text() == commit
    assert not home.state("models").exists()
    assert hub.located("thing-Q4_K_M.gguf") == link
    assert "moved" in home.state(mig.LOG).read_text()


def test_a_second_run_is_idempotent(old, server):
    assert run().moved == 2
    again = run()
    assert again.ok and again.files == 0 and again.moved == 0
    assert mig.command("run") == 0


def test_a_blob_the_cache_already_has_is_verified_and_ours_removed(old, server, tmp_path):
    hub.pull(f"hf:{REPO}/thing-Q4_K_M.gguf")
    blob = cache_root(tmp_path) / "blobs" / hashlib.sha256(BIG).hexdigest()
    before = blob.stat()
    report = run()
    assert report.ok and report.deduped == 1 and report.moved == 2
    assert not old["big"].exists() and blob.stat().st_ino == before.st_ino
    assert blob.read_bytes() == BIG


def test_a_cut_off_run_is_finished_by_the_next(old, server, tmp_path):
    blob = cache_root(tmp_path) / "blobs" / hashlib.sha256(BIG).hexdigest()
    blob.parent.mkdir(parents=True)
    os.link(old["big"], blob)            # the run died after linking and before removing
    report = run()
    assert report.ok and report.deduped == 1 and not old["big"].exists()


def test_a_blob_of_that_name_with_other_bytes_is_never_replaced(old, server, tmp_path):
    blob = cache_root(tmp_path) / "blobs" / hashlib.sha256(BIG).hexdigest()
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * len(BIG))
    report = run()
    assert not report.ok and old["big"].read_bytes() == BIG and blob.read_bytes() == b"x" * len(BIG)
    assert any("other bytes" in row for row in report.left)


def test_a_file_whose_hash_differs_from_the_hubs_is_refused_and_kept(old, server, tmp_path):
    bad = bytearray(BIG)
    bad[-1] ^= 1
    old["big"].write_bytes(bytes(bad))
    report = run()
    assert not report.ok and report.moved == 1
    assert old["big"].read_bytes() == bytes(bad)
    assert any("differs from the digest" in row for row in report.left)
    assert not (cache_root(tmp_path) / "blobs" / hashlib.sha256(bytes(bad)).hexdigest()).exists()
    assert not (cache_root(tmp_path) / "snapshots" / server.commit(REPO) / "thing-Q4_K_M.gguf").exists()


def test_a_file_of_the_wrong_size_is_left_and_listed(old, server):
    old["big"].write_bytes(BIG[:-5])
    report = run()
    assert old["big"].exists() and any("bytes, the Hub lists" in row for row in report.left)


def test_a_repository_the_hub_no_longer_has_is_left_in_place_and_listed(old, server):
    gone = old_file(GONE, "m.gguf", BIG)
    report = run()
    assert gone.read_bytes() == BIG and not report.ok
    assert any(str(gone) in row and GONE in row for row in report.left)
    assert report.moved == 2


def test_an_unreachable_hub_leaves_everything_where_it_is(old, server, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
    report = run()
    assert report.moved == 0 and len(report.left) == 2 and all(p.exists() for p in old.values())
    assert mig.command("run") == 1


def test_a_file_a_server_holds_blocks_the_whole_run(old, server, tmp_path):
    report = mig.migrate(apply=True, models=lambda: [str(old["big"])])
    assert report.blocked and "poolhouse-serve leases" in report.blocked[0]
    assert all(p.exists() for p in old.values()) and not (tmp_path / "hub").exists()
    assert mig.migrate(apply=False, models=lambda: [str(old["big"])]).blocked


def test_only_models_inside_the_old_store_count_as_held(old, tmp_path):
    root = home.state("models")
    assert mig.leased(root, [str(old["small"]), "hf:maker/x/y.gguf", str(tmp_path / "elsewhere.gguf")]) == [REPO]
    assert mig.leased(root, []) == []


def test_the_broker_default_finds_no_holders_on_a_machine_with_none():
    assert mig.broker_models() == []


def test_a_cache_that_cannot_be_written_refuses(old, server, tmp_path, monkeypatch):
    (tmp_path / "file").write_text("x")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "file" / "hub"))
    report = mig.migrate(apply=True, models=lambda: [])
    assert report.blocked and "cannot be written" in report.blocked[0]
    assert all(p.exists() for p in old.values())


def test_another_volume_copies_checks_then_removes(old, server, tmp_path, monkeypatch):
    real = os.link

    def across(src, dst, *a, **k):
        if str(dst).endswith(".incomplete"):
            return real(src, dst, *a, **k)
        if "blobs" in str(dst) and str(src) == str(old["big"]):
            raise OSError(errno.EXDEV, "cross-device link")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(mig.os, "link", across)
    report = run()
    blob = cache_root(tmp_path) / "blobs" / hashlib.sha256(BIG).hexdigest()
    assert report.ok and blob.read_bytes() == BIG and not old["big"].exists()
    assert not list((tmp_path / "hub").rglob("*.incomplete"))


def test_stray_and_half_written_files_are_never_moved(old, server):
    stray = home.state("models") / "loose.gguf"
    stray.write_bytes(b"x")
    half = old_file(REPO, "later.gguf.part", b"x")
    report = run()
    assert stray.exists() and half.exists() and report.moved == 2
    assert sum("not a finished file" in row for row in report.left) == 2


def test_the_command_prints_json_and_exits_by_the_outcome(old, server, capsys):
    assert models_cli.run(["migrate", "plan", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["files"] == 2
    assert models_cli.run(["migrate", "run"]) == 0
    assert "moved       2 file(s)" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        models_cli.run(["migrate", "now"])
