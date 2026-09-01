"""A JSON file is rewritten whole, or not at all."""

import json

import pytest

from ml_stack.files import prune_orphans, read_json, write_json


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
