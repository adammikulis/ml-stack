"""The session guard over the real ml_stack cache."""

from __future__ import annotations

import json
import os

from conftest import changed_files, file_mtimes, truncated_logs

LOG = "llama-server-8080.log"


def test_a_log_that_grew_is_a_live_server():
    assert truncated_logs({LOG: (7, 10)}, {LOG: (7, 99)}) == []


def test_the_same_file_getting_shorter_is_a_truncation():
    assert truncated_logs({LOG: (7, 99)}, {LOG: (7, 10)}) == [LOG]


def test_a_restart_writes_a_new_file_under_the_same_name():
    assert truncated_logs({LOG: (7, 99)}, {LOG: (8, 10)}) == []


def test_a_new_log_is_another_server_starting():
    assert truncated_logs({}, {"llama-server-50085.log": (9, 4096)}) == []


def test_a_log_that_went_away_is_not_a_truncation():
    assert truncated_logs({LOG: (7, 99)}, {}) == []


def test_only_external_keystore_lock_owners_are_excluded(tmp_path, monkeypatch):
    import conftest

    root = tmp_path / "keystore"
    root.mkdir()
    monkeypatch.setattr(conftest, 'ours', lambda entry: entry['owner_pid'] == os.getpid())
    (root / 'state.lock').write_text('12345')
    (root / 'flight.lock').write_text('12345')
    assert file_mtimes(tmp_path) == {}
    (root / 'state.lock').write_text(str(os.getpid()))
    assert changed_files({}, file_mtimes(tmp_path)) == ['keystore/state.lock']
    (root / 'flight.lock').write_text('invalid owner')
    assert set(file_mtimes(tmp_path)) == {'keystore/state.lock', 'keystore/flight.lock'}


def test_keystore_records_keys_and_unknown_locks_remain_guarded(tmp_path):
    root = tmp_path / 'keystore'
    root.mkdir()
    for name in ('state.json', 'master.key', 'credentials.json', 'other.lock'):
        (root / name).write_text('12345')
    assert set(file_mtimes(tmp_path)) == {f'keystore/{name}' for name in
        ('state.json', 'master.key', 'credentials.json', 'other.lock')}


def test_keystore_rate_metadata_distinguishes_test_and_external_writes(tmp_path, monkeypatch):
    import conftest

    root = tmp_path / 'keystore'
    root.mkdir()
    path = root / 'rate.json'
    monkeypatch.setattr(conftest, 'ours', lambda entry: entry['owner_pid'] == os.getpid())
    path.write_text(json.dumps({'writer_pid': 12345, 'reads': [], 'writes': []}))
    assert file_mtimes(tmp_path) == {}
    path.write_text(json.dumps({'writer_pid': os.getpid(), 'reads': [], 'writes': []}))
    assert changed_files({}, file_mtimes(tmp_path)) == ['keystore/rate.json']
    path.write_text(json.dumps({'reads': [], 'writes': []}))
    assert 'keystore/rate.json' in file_mtimes(tmp_path)


def test_keystore_spending_records_actual_pid_without_calling_backend(tmp_path):
    from ml_stack.keystore import Keystore

    held = Keystore(directory=tmp_path / 'ks')
    held._spend('read', 'process provenance')
    record = json.loads((tmp_path / 'ks' / 'rate.json').read_text())
    assert record['writer_pid'] == os.getpid()
    assert record['writer_at'] == record['reads'][-1]


def test_only_live_harness_keys_linked_to_external_worker_are_excluded(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import conftest
    import psutil

    worker = tmp_path / 'workspace' / 'local-agents'
    worker.mkdir(parents=True)
    (worker / 'scout.json').write_text(json.dumps({'pid': 12345, 'process_started': 10.0}))
    key = worker / 'scout-chats' / 'harness' / ('a' * 12) / ('b' * 24) / 'sessions' / ('23456.' + 'c' * 64 + '.key')
    key.parent.mkdir(parents=True)
    key.write_bytes(b'private test session key')
    parent = SimpleNamespace(pid=12345, create_time=lambda: 10.0)
    monkeypatch.setattr(conftest, 'ours', lambda entry: entry['owner_pid'] == os.getpid())
    monkeypatch.setattr(psutil, 'Process', lambda pid: SimpleNamespace(parents=lambda: [parent]))
    assert key.relative_to(tmp_path).as_posix() not in file_mtimes(tmp_path)
    (worker / 'scout.json').write_text(json.dumps({'pid': 12345, 'process_started': 9.0}))
    assert key.relative_to(tmp_path).as_posix() in file_mtimes(tmp_path)
    malformed = key.with_name('23456.bad.key')
    malformed.write_bytes(b'private test session key')
    assert malformed.relative_to(tmp_path).as_posix() in file_mtimes(tmp_path)
    (worker / 'scout.json').write_text(json.dumps({'pid': 12345, 'process_started': 10.0}))
    ours_key = key.with_name(str(os.getpid()) + '.' + 'c' * 64 + '.key')
    ours_key.write_bytes(b'private test session key')
    assert ours_key.relative_to(tmp_path).as_posix() in file_mtimes(tmp_path)


def test_scanner_write_attribution_excludes_only_live_external_registered_process(tmp_path, monkeypatch):
    import conftest
    import psutil

    sentinel = tmp_path / 'sentinel'
    sentinel.mkdir()
    now = conftest.time.time()
    payload = {'pid': 12345, 'started': now - 10, 'process_started': now - 20,
               'beat': now, 'interval_s': 60, 'running': True}
    class Process:
        def __init__(self, pid):
            self.pid = pid
        def create_time(self):
            return now - 20
        def is_running(self):
            return True
    monkeypatch.setattr(psutil, 'Process', Process)
    monkeypatch.setattr(conftest, 'ours', lambda entry: entry['owner_pid'] == os.getpid())
    def save():
        (sentinel / 'scanner.json').write_text(json.dumps({'payload': payload}))
        (sentinel / 'scanner.json.prev').write_text(json.dumps({'payload': payload}))
        (sentinel / 'state.lock').write_text(str(payload['pid']))
    save()
    assert file_mtimes(tmp_path) == {}
    payload['pid'] = os.getpid()
    save()
    assert len(file_mtimes(tmp_path)) == 3
    payload.update(pid=12345, process_started=now - 30)
    save()
    assert len(file_mtimes(tmp_path)) == 3
    payload.pop('pid')
    (sentinel / 'scanner.json').write_text(json.dumps({'payload': payload}))
    assert len(file_mtimes(tmp_path)) == 3
    payload.update(pid=12345, process_started=now - 20)
    save()
    monkeypatch.setattr(Process, 'is_running', lambda self: False)
    assert len(file_mtimes(tmp_path)) == 3
    monkeypatch.setattr(Process, 'is_running', lambda self: True)
    monkeypatch.setattr(Process, 'create_time', lambda self: now + 1)
    assert len(file_mtimes(tmp_path)) == 3
