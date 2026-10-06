"""MedMCQA conversion selects labelled single-answer cases."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("prepare_medmcqa", Path(__file__).parents[1] / "scripts/prepare_medmcqa.py")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def row(index=0):
    return {"choice_type": "single", "question": "  An invented exam question?  ",
            "opa": " first ", "opb": "second", "opc": "third", "opd": "fourth",
            "cop": index % 4, "id": f"case-{index}", "exp": "unused"}


def test_labels_options_and_groups_preserve_case_identity():
    case = converter.convert(row(2))
    assert case == {"id": "case-2", "group": "case-2", "kind": "choice",
                    "question": "Which option correctly answers this medical exam question?",
                    "state": "An invented exam question?",
                    "options": {"A": "first", "B": "second", "C": "third", "D": "fourth"},
                    "label": "C"}


@pytest.mark.parametrize("changes", [{"choice_type": "multi"}, {"cop": True}, {"cop": -1},
                                      {"cop": 4}, {"cop": None}, {"opa": " "},
                                      {"question": None}, {"id": ""}])
def test_unsupported_and_unlabelled_cases_are_omitted(changes):
    assert converter.convert({**row(), **changes}) is None


def parquet(monkeypatch, rows):
    package = ModuleType("pyarrow")
    module = ModuleType("pyarrow.parquet")
    module.read_table = lambda path: SimpleNamespace(to_pylist=lambda: rows)
    package.parquet = module
    monkeypatch.setitem(sys.modules, "pyarrow", package)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", module)


def test_split_hash_matches_written_jsonl_and_unsupported_rows_are_removed(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/train-0.parquet").touch()
    parquet(monkeypatch, [*(row(index) for index in range(40)), {**row(), "choice_type": "multi"}])
    output = tmp_path / "train.jsonl"
    count, digest = converter.convert_split(tmp_path, "train", output)
    assert count == 40 and digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert len({json.loads(line)["id"] for line in output.read_text().splitlines()}) == 40


def test_small_or_missing_split_is_refused(tmp_path, monkeypatch):
    output = tmp_path / "train.jsonl"
    parquet(monkeypatch, [row()])
    with pytest.raises(FileNotFoundError):
        converter.convert_split(tmp_path, "train", output)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/train-0.parquet").touch()
    with pytest.raises(ValueError, match="only 1 supported cases"):
        converter.convert_split(tmp_path, "train", output)
    assert not output.exists()


def test_cli_refuses_generated_cases_in_checkout(tmp_path, monkeypatch):
    snapshot = tmp_path / converter.REVISION
    snapshot.mkdir()
    (snapshot / "README.md").touch()
    output = tmp_path / "checkout" / "cases"
    monkeypatch.setattr(sys, "argv", ["prepare_medmcqa", str(snapshot), str(output)])
    monkeypatch.setattr(converter.worktreerules, "checkouts", lambda path: (tmp_path, tmp_path))
    with pytest.raises(SystemExit) as failure:
        converter.main()
    assert failure.value.code == 2 and not output.exists()


def test_split_refuses_output_inside_nested_checkout(tmp_path, monkeypatch):
    output = tmp_path / "nested" / "train.jsonl"
    monkeypatch.setattr(converter.worktreerules, "checkouts", lambda path: (tmp_path, tmp_path))
    with pytest.raises(ValueError, match="outside a Git checkout"):
        converter.convert_split(tmp_path, "train", output)
    assert not output.exists()


def test_split_refuses_link_target_before_truncation(tmp_path, monkeypatch):
    from test_hub_discover import symlink
    target = tmp_path / "valuable.jsonl"
    target.write_bytes(b"preserve")
    output = tmp_path / "cases" / "train.jsonl"
    output.parent.mkdir()
    symlink(output, target)
    with pytest.raises(ValueError):
        converter.convert_split(tmp_path, "train", output)
    assert target.read_bytes() == b"preserve"


def test_atomic_output_preserves_existing_hardlink_target(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/train-0.parquet").touch()
    parquet(monkeypatch, [row(index) for index in range(40)])
    target = tmp_path / "valuable.jsonl"
    target.write_bytes(b"preserve")
    output = tmp_path / "train.jsonl"
    output.hardlink_to(target)
    converter.convert_split(tmp_path, "train", output)
    assert target.read_bytes() == b"preserve" and output.read_bytes() != b"preserve"
