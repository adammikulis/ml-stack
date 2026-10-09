"""A cached test artifact is built once per source state and rebuilt when a source changes."""

from __future__ import annotations

import os
import tempfile

import artifact_cache


def test_an_artifact_is_built_once_for_the_same_sources_and_again_for_a_changed_one(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    source = tmp_path / "source.py"
    source.write_text("one\n")
    built = []

    def build(into):
        built.append(into)
        (into / "out.txt").write_text("made\n")

    first = artifact_cache.cached("demo", [source], build)
    again = artifact_cache.cached("demo", [source], build)
    assert first == again and len(built) == 1 and (first / "out.txt").read_text() == "made\n"

    source.write_text("two, longer\n")
    os.utime(source, ns=(1, 1))
    changed = artifact_cache.cached("demo", [source], build)
    assert changed != first and len(built) == 2


def test_a_worker_that_loses_the_race_discards_its_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    source = tmp_path / "source.py"
    source.write_text("one\n")

    def build(into):
        # another worker publishes the finished directory while this one is still building
        published = into.parent / artifact_cache.fingerprint([source])
        published.mkdir()
        (published / "theirs.txt").write_text("theirs")
        (into / "mine.txt").write_text("mine")

    got = artifact_cache.cached("race", [source], build)
    assert (got / "theirs.txt").exists() and not (got / "mine.txt").exists()
    assert sorted(p.name for p in got.parent.iterdir()) == [got.name]
