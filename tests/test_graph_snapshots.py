"""Copies of a store you can actually go back to.

Every test works on real files. Where the claim is "the copy holds what the original held",
the copy is opened again to find out, because that is the whole point of the module.
"""

import json
from pathlib import Path

import pytest

from ml_stack.graph.snapshots import (Snapshot, SnapshotError, clone_file, prune, read_manifest,
                                      remove_store, restore, snapshots, take, unmanaged)


def counter(path):
    """What is 'in' a store, for a store that is just a text file."""
    body = Path(path).read_text(encoding="utf-8")
    return {"lines": len([x for x in body.splitlines() if x.strip()])}


def a_store(tmp_path, lines=3, name="g.store"):
    path = tmp_path / name
    path.write_text("\n".join(f"row {i}" for i in range(lines)), encoding="utf-8")
    return path


def test_a_snapshot_is_verified_by_opening_it(tmp_path):
    src = a_store(tmp_path, 5)
    record = take(src, reason="before a migration", count=counter)
    assert record.counts == {"lines": 5}
    assert record.reason == "before a migration"
    assert Path(record.path).exists() and Path(record.path).parent.name == "_backups"
    assert counter(record.path) == {"lines": 5}
    assert record.method in ("clonefile", "copy")


def test_a_clone_that_does_not_match_is_thrown_away(tmp_path):
    src = a_store(tmp_path, 4)
    calls = []

    def lying(path):
        calls.append(Path(path))
        # the source counts 4; the clone will claim 1
        return {"lines": 4 if len(calls) == 1 else 1}

    with pytest.raises(SnapshotError, match="does not match the source"):
        take(src, reason="doomed", count=lying)
    assert list((tmp_path / "_backups").glob("*.store")) == []


def test_a_clone_that_will_not_open_is_thrown_away(tmp_path):
    src = a_store(tmp_path)
    seen = []

    def breaks(path):
        seen.append(path)
        if len(seen) > 1:
            raise ValueError("unreadable")
        return {"lines": 3}

    with pytest.raises(SnapshotError, match="would not open"):
        take(src, reason="doomed", count=breaks)
    assert list((tmp_path / "_backups").glob("*.store")) == []


def test_the_log_comes_too_and_is_folded_in(tmp_path):
    src = a_store(tmp_path)
    Path(str(src) + ".wal").write_text("uncheckpointed", encoding="utf-8")
    folded = []
    take(src, reason="with a log", count=counter, fold=lambda p: folded.append(Path(p)))
    assert folded, "the log was copied but never folded in"
    assert (tmp_path / "_backups").exists()


def test_a_fold_that_fails_means_the_source_is_not_safely_copyable(tmp_path):
    src = a_store(tmp_path)
    Path(str(src) + ".wal").write_text("uncheckpointed", encoding="utf-8")

    def refuses(_):
        raise RuntimeError("cannot open")

    with pytest.raises(SnapshotError, match="never checkpointed"):
        take(src, reason="doomed", count=counter, fold=refuses)


def test_two_snapshots_in_one_second_do_not_overwrite_each_other(tmp_path):
    src = a_store(tmp_path)
    first = take(src, reason="one", count=counter)
    second = take(src, reason="two", count=counter)
    assert first.path != second.path
    assert len(snapshots(src)) == 2


def test_only_the_newest_are_kept(tmp_path):
    src = a_store(tmp_path)
    for i in range(5):
        take(src, reason=f"take {i}", count=counter, keep=3)
    kept = snapshots(src)
    assert len(kept) == 3
    assert [r.reason for r in kept] == ["take 4", "take 3", "take 2"]
    # the manifests of the pruned ones went too
    assert len(list((tmp_path / "_backups").glob("*.json"))) == 3


def test_restoring_puts_it_back_and_keeps_what_was_there(tmp_path):
    src = a_store(tmp_path, 3)
    record = take(src, reason="known good", count=counter)
    src.write_text("row 0\nrow 1\nrow 2\nrow 3\nrow 4\nBROKEN", encoding="utf-8")
    assert counter(src)["lines"] == 6

    restore(record.path, count=counter)
    assert counter(src) == {"lines": 3}
    # the state before the restore was kept, so the restore is itself undoable
    assert any("before restoring" in r.reason for r in snapshots(src))
    # and the snapshot survives for a second go
    assert Path(record.path).exists()


def test_a_snapshot_that_changed_since_it_was_taken_is_refused(tmp_path):
    src = a_store(tmp_path, 3)
    record = take(src, reason="known good", count=counter)
    Path(record.path).write_text("row 0", encoding="utf-8")
    with pytest.raises(SnapshotError, match="no longer matches its manifest"):
        restore(record.path, count=counter)


def test_a_file_this_cannot_identify_is_not_restored(tmp_path):
    stray = tmp_path / "_backups" / "someone-elses.store"
    stray.parent.mkdir(parents=True)
    stray.write_text("who knows", encoding="utf-8")
    with pytest.raises(SnapshotError, match="no readable manifest"):
        restore(stray, count=counter)
    assert read_manifest(stray) is None
    assert unmanaged(tmp_path / "g.store") == [stray]


def test_removing_a_store_takes_its_log(tmp_path):
    src = a_store(tmp_path)
    wal = Path(str(src) + ".wal")
    wal.write_text("tail", encoding="utf-8")
    remove_store(src)
    assert not src.exists() and not wal.exists()


def test_a_clone_shares_blocks_where_it_can(tmp_path):
    src = a_store(tmp_path)
    dst = tmp_path / "copy.store"
    method = clone_file(src, dst)
    assert method in ("clonefile", "copy")
    assert dst.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def test_a_manifest_that_is_not_a_snapshot_is_not_read_as_one(tmp_path):
    path = tmp_path / "g.store"
    path.write_text("x", encoding="utf-8")
    path.with_suffix(".store.json").write_text(json.dumps({"nonsense": 1}), encoding="utf-8")
    assert read_manifest(path) is None
