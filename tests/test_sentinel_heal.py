"""What heals without asking anybody: a pinned file that is gone, a server that exited, and the
decider trainer's pins when a run dies."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from poolhouse import home, sentinel
from poolhouse.sentinel import core, human
from poolhouse.sentinel.store import State

TESTS = str(Path(__file__).resolve().parent)


def pinned(node, count=2, kind="model"):
    folder = home.home() / "runs" / "v1"
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for n in range(count):
        path = folder / f"part{n}.safetensors"
        path.write_bytes(b"weights %d" % n)
        node.manifest.pin(path, kind, source="trained:test")
        files.append(path)
    return files


def test_a_gone_pinned_file_is_logged_once_and_its_pin_dropped_after_three_scans():
    node = sentinel.default()
    node.heads_up.spawn = lambda work: pytest.fail("a dialog was asked for")
    files = pinned(node)
    keep, gone = files
    gone.unlink()
    for scan in range(1, core.GONE_SCANS):
        assert [f.event.kind for f in node.scan(deep=True)] == ["integrity.missing"]
        assert len(node.manifest.pins()) == 2, scan
    node.scan(deep=True)
    assert list(node.manifest.pins()) == [str(keep)]
    kinds = [e.kind for e in node.bus.log.read()]
    assert kinds.count("integrity.missing") == 1 and kinds.count("integrity.pin_dropped") == 1
    assert "quarantine.move_refused" not in kinds and "quarantine.quarantined" not in kinds
    assert node.store.records() == []
    assert node.chip()["verdict"] == "green"


def test_a_pinned_file_that_comes_back_resets_the_count():
    node = sentinel.default()
    (path,) = pinned(node, 1)
    data = path.read_bytes()
    path.unlink()
    node.scan(deep=True)
    node.scan(deep=True)
    path.write_bytes(data)
    node.scan(deep=True)
    path.unlink()
    node.scan(deep=True)
    node.scan(deep=True)
    assert str(path) in node.manifest.pins()


def test_loading_a_gone_pinned_file_is_refused_without_quarantining_it():
    node = sentinel.default()
    (path,) = pinned(node, 1)
    path.unlink()
    assert node.verify_before_load(path) is False
    assert node.store.records() == []


def test_a_changed_pinned_file_is_still_quarantined():
    node = sentinel.default()
    node.heads_up.spawn = lambda work: None
    (path,) = pinned(node, 1)
    path.write_bytes(b"tampered")
    node.scan(deep=True)
    assert [r.state for r in node.store.records()] == [State.QUARANTINED]
    assert not path.exists()


def test_a_record_an_older_version_made_for_a_gone_file_settles_on_the_next_scan():
    node = sentinel.default()
    (path,) = pinned(node, 1)
    path.unlink()
    record = node.store.quarantine(("model", str(path)), "integrity.missing: pinned_bytes=9", {})
    assert record.state == State.QUARANTINED
    node.scan(deep=True)
    assert node.store.get(record.id).state == State.RELEASED
    assert str(path) not in node.manifest.pins()


def test_only_a_missing_record_with_nothing_moved_settles():
    node = sentinel.default()
    present = home.home() / "present.bin"
    present.parent.mkdir(parents=True, exist_ok=True)
    present.write_bytes(b"x")
    cases = {
        "still there": ("model", str(present), "integrity.missing: x"),
        "other finding": ("model", str(home.home() / "gone-a.bin"), "integrity.content_changed: x"),
        "other kind": ("peer", str(home.home() / "gone-b.bin"), "integrity.missing: x"),
    }
    for kind, key, reason in cases.values():
        made = node.store.quarantine((kind, key), reason, {})
        assert node.store.settle_missing(made.id) is False
        assert node.store.get(made.id).state == State.QUARANTINED


def test_a_watched_server_whose_process_exited_clears_itself():
    node = sentinel.default()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        node.store.watch("server", "port:1", "server.unmanaged: x", {"pid": dead.pid})
        node.store.watch("server", "port:2", "server.unmanaged: x", {"pid": live.pid})
        node.scan()
        assert node.store.state_of("server", "port:1") == State.CLEAR
        assert node.store.state_of("server", "port:2") == State.WATCH
    finally:
        live.kill()
        live.wait()


# -- the decider trainer ----------------------------------------------------------------
KILLED = """
import sys, time
sys.path.insert(0, {tests!r})
from pathlib import Path
import poolhouse.train.decider as d
from test_train_decider import tiny_cases
from poolhouse.train.lora import Lora

def stall(*a, **k):
    Path({marker!r}).write_text("fitting")
    time.sleep(300)

d.fit_model = stall
settings = d.Settings(name="killed", base=Path({base!r}), steps=4, batch_size=4, device="cpu",
                      dtype="float32", lora=Lora(4, 8, 0.0, ("q_proj", "v_proj")), lr=1e-3,
                      test_fraction=0.25, calibrate_fraction=0.25, baseline="none")
d.train(tiny_cases(), Path({out!r}), settings)
"""


def test_a_decider_run_killed_while_fitting_leaves_no_pins(tmp_path):
    from test_train_decider import tiny_base
    base = tiny_base.__wrapped__(tmp_path)
    marker, out = tmp_path / "marker", tmp_path / "decider"
    script = KILLED.format(tests=TESTS, marker=str(marker), base=str(base), out=str(out))
    child = subprocess.Popen([sys.executable, "-c", script], env={
        **os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    try:
        for _ in range(600):
            if marker.exists() or child.poll() is not None:
                break
            time.sleep(0.1)
        assert marker.exists(), "the run never reached the fit"
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
    assert sentinel.default().manifest.pins() == {}


def test_pins_made_before_a_failure_part_way_are_dropped(tmp_path, monkeypatch):
    from poolhouse.sentinel.integrity import Manifest
    from poolhouse.train import decider
    root = tmp_path / "out"
    (root / "lora").mkdir(parents=True)
    for rel, _ in decider.PINNED:
        (root / rel).write_text("x")
    real, calls = Manifest.pin, []

    def flaky(self, path, kind, source=""):
        calls.append(path)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real(self, path, kind, source)

    monkeypatch.setattr(Manifest, "pin", flaky)
    with pytest.raises(KeyboardInterrupt):
        decider.pin_files(root, "x")
    assert len(calls) == 3
    assert sentinel.default().manifest.pins() == {}


def test_a_release_by_dialog_needs_no_terminal_but_an_agent_marker_refuses_it():
    grant = human.mint_clicked("release", "q-1", answer="Release", label="Release", env={})
    grant.check("release", "q-1")
    with pytest.raises(human.HumanRequired):
        grant.check("release", "q-2")
    with pytest.raises(human.HumanRequired):
        human.mint_clicked("release", "q-1", answer="Release", label="Release",
                           env={"CLAUDECODE": "1"})


def test_unchanged_deep_scan_preserves_manifest_and_state_lock_mtimes():
    node = sentinel.default()
    (path,) = pinned(node, 1)
    node.manifest.pin(path, "model")
    node.store.quarantine(("peer", "fixture-peer"), "fixture finding", {})
    names = ("manifest.json", "manifest.json.prev", "state.lock")
    before = {name: (node.root / name).stat().st_mtime_ns for name in names}
    node.scan(deep=True)
    assert {name: (node.root / name).stat().st_mtime_ns for name in names} == before


def test_deep_scan_records_changed_verified_metadata():
    node = sentinel.default()
    (path,) = pinned(node, 1)
    old = node.manifest.pins()[str(path)]
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1))
    assert node.scan(deep=True) == []
    refreshed = node.manifest.pins()[str(path)]
    assert refreshed.mtime_ns == old.mtime_ns + 1
    assert refreshed.sha256 == old.sha256


@pytest.mark.parametrize("operation", ["repin", "unpin"])
def test_metadata_refresh_preserves_concurrent_pin_change(operation, monkeypatch):
    from contextlib import contextmanager

    from poolhouse.sentinel import integrity

    node = sentinel.default()
    (path,) = pinned(node, 1)
    old = node.manifest.pins()[str(path)]
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1))
    take_lock, changed = integrity.only_one, []

    @contextmanager
    def concurrent_pin(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(integrity, "only_one", take_lock)
            if operation == "repin":
                changed.append(node.manifest.pin(path, "model", source="concurrent"))
            else:
                assert node.manifest.unpin(path)
        with take_lock(*args, **kwargs) as held:
            yield held

    monkeypatch.setattr(integrity, "only_one", concurrent_pin)
    node.manifest.refresh_stat(old)
    assert node.manifest.pins() == ({str(path): changed[0]} if operation == "repin" else {})


def test_settlement_rechecks_file_absence_after_lock(monkeypatch):
    from contextlib import contextmanager

    node = sentinel.default()
    (path,) = pinned(node, 1)
    path.unlink()
    record = node.store.quarantine(("model", str(path)), "integrity.missing: fixture", {})
    take_lock = node.store._locked

    @contextmanager
    def restored_before_lock():
        path.write_bytes(b"restored")
        with take_lock() as held:
            yield held

    monkeypatch.setattr(node.store, "_locked", restored_before_lock)
    assert node.store.settle_missing(record.id) is False
    assert node.store.get(record.id).state == State.QUARANTINED
