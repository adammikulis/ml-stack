"""Broker handshakes report observed runtime identity without guessing older processes."""

import os
import sys
from pathlib import Path

import psutil
import pytest

import ml_stack
from ml_stack.files import read_json
from ml_stack.net import git
from ml_stack.serve import broker_runtime, broker_wire


def test_runtime_source_tracks_loaded_package_not_callers_worktree(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    snapshot = broker_runtime.snapshot()
    source = Path(ml_stack.__file__).resolve().parents[2]
    assert snapshot['source_commit'] == git.head(source)
    assert snapshot['pid'] == os.getpid()
    assert snapshot['pid_started'] == psutil.Process().create_time()
    assert snapshot['owner'] == psutil.Process().username()
    assert snapshot['environment']['interpreter'] == sys.executable
    assert snapshot['environment']['prefix'] == sys.prefix
    assert len(snapshot['implementation_sha256']) == 64


def test_installed_package_does_not_claim_an_unrelated_git_commit(monkeypatch, tmp_path):
    package = tmp_path / 'site' / 'ml_stack'
    (package / 'serve').mkdir(parents=True)
    (package / 'serve' / 'broker_wire.py').write_text('installed implementation')
    monkeypatch.setattr(ml_stack, '__file__', str(package / '__init__.py'))
    monkeypatch.setattr(git, 'head', lambda *_: pytest.fail('unrelated repository lookup'))
    snapshot = broker_runtime.snapshot()
    assert snapshot['source_commit'] is None and snapshot['source_dirty'] is None
    assert snapshot['environment']['package_root'] == str(package)


def test_old_broker_status_exposes_unknown_without_losing_servers(monkeypatch):
    monkeypatch.setattr(broker_wire, 'call', lambda *args, **kwargs: {'ok': True, 'servers': [{'port': 51548}]})
    status = broker_wire.status()
    assert status['servers'] == [{'port': 51548}]
    assert status['runtime']['state'] == 'unknown'
    assert status['runtime']['source_commit'] is None
    assert status['runtime']['environment'] is None
    assert status['runtime']['compatibility'] == 'unknown'


def test_explicit_protocol_mismatch_has_an_actionable_owned_upgrade_notice():
    value = broker_runtime.reported({'protocol': broker_runtime.PROTOCOL + 1, 'pid': 123})
    assert value['compatibility'] == 'incompatible'
    assert 'owned broker' in value['action'] and 'foreign holders' in value['action']
    assert broker_runtime.reported({'protocol': True})['state'] == 'unknown'


def test_a_reused_record_pid_or_socket_cannot_claim_the_brokers_identity(monkeypatch):
    monkeypatch.setattr(broker_wire, '_record', lambda: {'pid': 123, 'pid_started': 45, 'port': 1})
    monkeypatch.setattr(broker_wire, 'pid_exists', lambda _: True)
    monkeypatch.setattr(broker_wire, 'started_at', lambda _: 46)
    monkeypatch.setattr(broker_wire, '_send', lambda *_args, **_kwargs: pytest.fail('reused PID contacted'))
    assert broker_wire._answering() is None
    monkeypatch.setattr(broker_wire, 'started_at', lambda _: 45)
    monkeypatch.setattr(broker_wire, '_send', lambda *_args, **_kwargs: {'ok': True, 'pid': 999})
    assert broker_wire._answering() is None


def test_managed_broker_advertises_the_same_durable_registration_and_handshake():
    record = None
    try:
        ping = broker_wire.call('ping')
        record = read_json(broker_wire.record_path(), {})
        assert record['runtime'] == ping['runtime']
        assert record['pid_started'] == ping['runtime']['pid_started']
        assert ping['runtime']['protocol'] == broker_runtime.PROTOCOL
        assert ping['runtime']['source_commit']
        assert 'token' not in ping['runtime']
        status = broker_wire.status()
        assert status['runtime']['compatibility'] == 'compatible'
        assert status['runtime']['pid'] == record['pid']
    finally:
        if record:
            process = psutil.Process(record['pid'])
            assert process.create_time() == record['pid_started']
            process.terminate()
            process.wait(10)


def test_known_incompatible_broker_refuses_mutation_but_allows_status(monkeypatch):
    record = {'runtime': {'protocol': broker_runtime.PROTOCOL + 1}}
    monkeypatch.setattr(broker_wire, '_reach', lambda **_: record)
    calls = []
    monkeypatch.setattr(broker_wire, '_send', lambda record, body, **kwargs: calls.append(body['op']) or {'ok': True})
    with pytest.raises(broker_wire.BrokerError, match='owned broker'):
        broker_wire.call('release', start=False, lease='foreign')
    assert calls == []
    assert broker_wire.status()['runtime']['state'] == 'unknown'
    assert calls == ['status']
