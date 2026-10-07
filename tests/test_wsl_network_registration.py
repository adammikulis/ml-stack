"""Same-user live-daemon discovery registration and descriptor refusals."""

import json
import os
import stat
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ml_stack.fleet import discovery, wsl_network, wsl_registration as registration


@pytest.fixture
def descriptor(tmp_path, monkeypatch):
    path = tmp_path / 'runtime' / 'windows-network.json'
    monkeypatch.setattr(registration, '_path', lambda: path)
    monkeypatch.setattr(registration, '_supported', lambda: True)
    monkeypatch.setattr(os, 'geteuid', lambda: 123, raising=False)
    monkeypatch.setattr(os, 'O_NOFOLLOW', getattr(os, 'O_NOFOLLOW', 0), raising=False)
    monkeypatch.setattr(registration, '_private', lambda *_args, **_kwargs: None)
    monkeypatch.delenv(registration.ENV, raising=False)
    process = MagicMock()
    process.create_time.return_value = 42.5
    process.is_running.return_value = True
    process.uids.return_value = SimpleNamespace(effective=123)
    process.cmdline.return_value = ['python', '-m', 'ml_stack.cli.wsl_daemon']
    monkeypatch.setattr(registration.psutil, 'Process', lambda *_args: process)
    config = {'address': ['127.0.0.1', 9000], 'token': 'a' * 64}
    return path, process, config


def test_cli_without_inherited_environment_uses_current_daemon(descriptor, monkeypatch):
    path, _, config = descriptor
    monkeypatch.setenv(registration.ENV, json.dumps(config))
    with registration.registered():
        monkeypatch.delenv(registration.ENV)
        socket_factory = MagicMock()
        monkeypatch.setattr(wsl_network, 'DiscoverySocket', socket_factory)
        discovery._socket(bind=('', 0))
        socket_factory.assert_called_once_with({'broadcast': False, 'bind': ('', 0), 'group': None})
        assert registration.configuration() == config
    assert not path.exists()


@pytest.mark.parametrize('changed', ['started', 'uid', 'command', 'running'])
def test_reused_or_foreign_daemon_is_refused(descriptor, monkeypatch, changed):
    _, process, config = descriptor
    monkeypatch.setenv(registration.ENV, json.dumps(config))
    with registration.registered():
        monkeypatch.delenv(registration.ENV)
        if changed == 'started':
            process.create_time.return_value = 999
        elif changed == 'uid':
            process.uids.return_value = SimpleNamespace(effective=999)
        elif changed == 'command':
            process.cmdline.return_value = ['different-worker']
        else:
            process.is_running.return_value = False
        with pytest.raises(OSError, match='stale or invalid'):
            registration.configuration()


def test_cleanup_preserves_replacement_daemon(descriptor, monkeypatch):
    path, _, config = descriptor
    monkeypatch.setenv(registration.ENV, json.dumps(config))
    with registration.registered():
        replacement = json.loads(path.read_text())
        replacement['nonce'] = 'replacement'
        path.write_text(json.dumps(replacement))
    assert json.loads(path.read_text()) == replacement


def test_explicit_configuration_has_priority_and_is_validated(descriptor, monkeypatch):
    path, _, config = descriptor
    path.parent.mkdir()
    path.write_text('invalid registration')
    monkeypatch.setenv(registration.ENV, json.dumps(config))
    assert registration.configuration() == config
    monkeypatch.setenv(registration.ENV, '{}')
    with pytest.raises(OSError, match='Invalid'):
        registration.configuration()


@pytest.mark.parametrize('raw', ['broken', '[]', 'x' * (registration.LIMIT + 1)])
def test_malformed_descriptor_never_falls_back_to_native_udp(descriptor, raw):
    path, _, _ = descriptor
    path.parent.mkdir()
    path.write_text(raw)
    with pytest.raises(OSError):
        discovery._socket()


@pytest.mark.parametrize('mode,uid', [(stat.S_IFLNK | 0o600, 123), (stat.S_IFREG | 0o644, 123), (stat.S_IFREG | 0o600, 999)])
def test_unsafe_descriptor_is_refused(descriptor, monkeypatch, mode, uid):
    monkeypatch.undo()
    monkeypatch.setattr(os, 'geteuid', lambda: 123, raising=False)
    with pytest.raises(OSError, match='private and owned'):
        registration._private(SimpleNamespace(st_mode=mode, st_uid=uid))


def test_oversized_explicit_configuration_cannot_be_registered(descriptor, monkeypatch):
    path, _, config = descriptor
    config['unused'] = '界' * (registration.LIMIT // 2)
    monkeypatch.setenv(registration.ENV, json.dumps(config, ensure_ascii=False))
    with pytest.raises(OSError, match='Invalid'), registration.registered():
        pytest.fail('oversized bridge was registered')
    assert not path.exists()
