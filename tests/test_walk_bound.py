"""The real-state snapshot raises when it walks more than the limit."""

from __future__ import annotations

import pytest
import walkbound
from conftest import file_mtimes


def populate(root, layout):
    for rel in layout:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")


def test_a_walk_over_the_limit_fails_naming_each_top_level_directory(tmp_path, monkeypatch):
    populate(tmp_path, [f"big/{i}" for i in range(6)] + [f"small/{i}" for i in range(2)] + ["top"])
    monkeypatch.setattr(walkbound, "LIMIT", 5)
    with pytest.raises(walkbound.WalkTooLarge) as caught:
        file_mtimes(tmp_path)
    assert "big 6" in str(caught.value)
    assert "walked more than 5 files" in str(caught.value)


def test_a_walk_at_the_limit_passes(tmp_path, monkeypatch):
    populate(tmp_path, [f"d/{i}" for i in range(5)])
    monkeypatch.setattr(walkbound, "LIMIT", 5)
    assert len(file_mtimes(tmp_path)) == 5


def test_skipped_directories_do_not_count_toward_the_limit(tmp_path, monkeypatch):
    populate(tmp_path, [f"runtimes/{i}" for i in range(50)] + ["keep/a"])
    monkeypatch.setattr(walkbound, "LIMIT", 5)
    assert list(file_mtimes(tmp_path)) == ["keep/a"]
