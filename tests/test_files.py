"""A JSON file is rewritten whole, or not at all."""

import errno
import json

import pytest

from ml_stack import files as files_module
from ml_stack.files import (
    UNVERSIONED,
    CrossDevice,
    promote,
    prune_orphans,
    read_json,
    version_of,
    versioned,
    write_json,
    write_text,
    writing,
)


def test_what_was_written_is_read_back_and_nothing_is_left_beside_it(tmp_path):
    p = tmp_path / "state" / "graph.json"
    write_json(p, {"who": "Bea Marlow", "n": 1})
    write_json(p, {"who": "Bea Marlow", "n": 2})
    assert read_json(p, None) == {"who": "Bea Marlow", "n": 2}
    assert [f.name for f in p.parent.iterdir()] == ["graph.json"]


def test_a_write_that_fails_leaves_the_old_file_and_no_temporary(tmp_path):
    p = tmp_path / "graph.json"
    write_json(p, {"n": 1})
    with pytest.raises(TypeError):
        write_json(p, {"n": object()})
    assert json.loads(p.read_text()) == {"n": 1}
    assert [f.name for f in tmp_path.iterdir()] == ["graph.json"]


def test_a_missing_or_broken_file_reads_as_the_default(tmp_path):
    assert read_json(tmp_path / "absent.json", {"_v": 1}) == {"_v": 1}
    (tmp_path / "broken.json").write_text("{not json")
    assert read_json(tmp_path / "broken.json", []) == []


def test_text_is_kept_as_written_not_escaped(tmp_path):
    p = tmp_path / "t.json"
    write_json(p, {"place": "Zürich"})
    assert "Zürich" in p.read_text(encoding="utf-8")


def test_a_file_for_a_record_the_log_no_longer_has_is_deleted(tmp_path):
    d = tmp_path / "extractions"
    d.mkdir()
    (d / "C1-1.000001.json").write_text("{}")
    (d / "C1-9.999999.json").write_text("{}")
    (d / "notes.txt").write_text("not a record")
    assert prune_orphans(d, {"C1-1.000001"}) == ["C1-9.999999"]
    assert sorted(f.name for f in d.iterdir()) == ["C1-1.000001.json", "notes.txt"]


def test_a_directory_that_does_not_exist_has_no_orphans(tmp_path):
    assert prune_orphans(tmp_path / "nowhere", set()) == []


def test_a_finished_file_takes_the_targets_place_in_one_step(tmp_path):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"the old one")
    (tmp_path / "model.gguf.part").write_bytes(b"the new one")
    assert promote(tmp_path / "model.gguf.part", target) == target
    assert target.read_bytes() == b"the new one"
    assert [f.name for f in tmp_path.iterdir()] == ["model.gguf"]


def test_a_directory_and_a_symlink_are_promoted_the_same_way(tmp_path):
    staging = tmp_path / "staging"
    (staging / "weights").mkdir(parents=True)
    promote(staging, tmp_path / "checkpoint-100")
    assert (tmp_path / "checkpoint-100" / "weights").is_dir()

    (tmp_path / "latest.tmp").symlink_to("checkpoint-100", target_is_directory=True)
    promote(tmp_path / "latest.tmp", tmp_path / "latest")
    assert (tmp_path / "latest").is_symlink()
    assert (tmp_path / "latest" / "weights").is_dir()


def test_a_move_across_filesystems_says_so_rather_than_copying(tmp_path, monkeypatch):
    """A shared model cache is tens of gigabytes and the disk has them once, so the caller
    has to be able to tell this apart from any other OSError."""
    def elsewhere(_source, _target):
        raise OSError(errno.EXDEV, "Cross-device link")

    monkeypatch.setattr(files_module.os, "replace", elsewhere)
    (tmp_path / "cache").mkdir()
    with pytest.raises(CrossDevice, match="different filesystems"):
        promote(tmp_path / "cache", tmp_path / "shared")
    assert (tmp_path / "cache").is_dir()


def test_an_interrupted_write_leaves_the_previous_file_whole(tmp_path):
    p = tmp_path / "reads.json"
    p.write_text('{"1.1": {"unit": "1.1"}}')
    with pytest.raises(KeyboardInterrupt), writing(p) as tmp:
        tmp.write_text("half of the ne")
        raise KeyboardInterrupt("killed part-way through")
    assert json.loads(p.read_text()) == {"1.1": {"unit": "1.1"}}
    assert [f.name for f in tmp_path.iterdir()] == ["reads.json"]


def test_text_arrives_whole_or_not_at_all(tmp_path):
    p = tmp_path / "rows.jsonl"
    write_text(p, '{"id": 1}\n')
    assert p.read_text() == '{"id": 1}\n'
    assert [f.name for f in tmp_path.iterdir()] == ["rows.jsonl"]


def test_a_record_says_which_shape_it_has(tmp_path):
    p = tmp_path / "stamp.json"
    write_json(p, versioned({"url": "https://example.invalid/m.gguf"}, 1))
    assert read_json(p, None) == {"version": 1, "url": "https://example.invalid/m.gguf"}
    assert version_of(read_json(p, None)) == 1


def test_a_record_written_before_the_key_existed_reads_as_unversioned(tmp_path):
    p = tmp_path / "old.json"
    p.write_text('{"url": "https://example.invalid/m.gguf"}')
    assert version_of(read_json(p, None)) == UNVERSIONED
    assert version_of("not a record") == UNVERSIONED
    assert version_of({"version": "not a number"}) == UNVERSIONED


def test_sha256_file_matches_hashlib_across_chunk_boundaries(tmp_path):
    import hashlib

    from ml_stack.files import sha256_file

    data = bytes(range(256)) * 1000
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert sha256_file(path, chunk=4096) == sha256_file(path) == hashlib.sha256(data).hexdigest()
