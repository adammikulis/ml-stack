"""Reading, writing and compacting a JSON-lines file."""

import json

import pytest

from ml_stack.jsonl import MalformedLine, append, compact, read, ts_key, write


def _write(p, rows):
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _key(row):
    return row.get("id")


def test_the_last_line_per_key_wins(tmp_path):
    p = tmp_path / "log.jsonl"
    _write(p, [{"id": "a", "text": "first thoughts"},
               {"id": "b", "text": "something else"},
               {"id": "a", "text": "what was meant"}])
    assert compact(p, _key) == (2, 1)
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert [r["text"] for r in rows] == ["what was meant", "something else"]
    assert "first thoughts" not in p.read_text()


def test_output_keeps_the_order_keys_first_appeared(tmp_path):
    p = tmp_path / "log.jsonl"
    _write(p, [{"id": "b", "n": 1}, {"id": "a", "n": 1}, {"id": "b", "n": 2}])
    compact(p, _key)
    assert [json.loads(line)["id"] for line in p.read_text().splitlines()] == ["b", "a"]


def test_an_already_tidy_file_is_not_rewritten(tmp_path):
    p = tmp_path / "log.jsonl"
    _write(p, [{"id": "a", "text": "only"}])
    before = p.read_text()
    assert compact(p, _key) == (1, 0)
    assert p.read_text() == before


def test_dropped_keys_lose_every_line(tmp_path):
    p = tmp_path / "log.jsonl"
    _write(p, [{"id": "a", "n": 1}, {"id": "b", "n": 1}, {"id": "a", "n": 2}])
    assert compact(p, _key, drop={"a"}) == (1, 2)
    assert json.loads(p.read_text()) == {"id": "b", "n": 1}


def test_junk_lines_and_keyless_rows_are_dropped(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text('{"id": "a", "n": 1}\nnot json\n{"n": 2}\n')
    assert compact(p, _key) == (1, 2)
    assert [json.loads(line)["id"] for line in p.read_text().splitlines()] == ["a"]


def test_order_picks_the_greatest_not_the_last(tmp_path):
    p = tmp_path / "log.jsonl"
    _write(p, [{"id": "a", "v": 3}, {"id": "a", "v": 1}])
    compact(p, _key, order=lambda r: r["v"])
    assert json.loads(p.read_text())["v"] == 3


def test_order_ties_still_go_to_the_later_line(tmp_path):
    p = tmp_path / "log.jsonl"
    _write(p, [{"id": "a", "v": 1, "text": "old"}, {"id": "a", "v": 1, "text": "new"}])
    compact(p, _key, order=lambda r: r["v"])
    assert json.loads(p.read_text())["text"] == "new"


def test_a_missing_file_compacts_to_nothing(tmp_path):
    assert compact(tmp_path / "absent.jsonl", _key) == (0, 0)


def test_ts_key_compares_across_digit_counts():
    assert ts_key("2.000010") < ts_key("10.000001")
    assert ts_key("5") == (5, 0)
    assert ts_key("not a number") is None


def test_a_file_written_reads_back_as_the_records_that_went_in(tmp_path):
    p = tmp_path / "deep" / "rows.jsonl"
    assert write(p, [{"a": 1}, {"note": "caf\u00e9"}]) == 2
    assert append(p, [{"a": 2}]) == 1
    assert read(p) == [{"a": 1}, {"note": "caf\u00e9"}, {"a": 2}]
    assert "caf\u00e9" in p.read_text(encoding="utf-8"), "non-ASCII is written as itself"


def test_writing_replaces_rather_than_appends(tmp_path):
    p = tmp_path / "rows.jsonl"
    write(p, [{"a": 1}])
    write(p, [{"a": 2}])
    assert read(p) == [{"a": 2}]


def test_a_line_that_is_not_json_is_named_rather_than_skipped(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(MalformedLine, match="rows.jsonl:2"):
        read(p)


def test_a_type_json_has_no_shape_for_is_written_by_default(tmp_path):
    p = tmp_path / "rows.jsonl"
    write(p, [{"when": tmp_path}], default=str)
    assert read(p) == [{"when": str(tmp_path)}]
