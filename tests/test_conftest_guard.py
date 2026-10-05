"""The session guard over the real ml_stack cache."""

from __future__ import annotations

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
