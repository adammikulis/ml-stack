"""Private runtime discovery registration for Windows-launched WSL daemons."""

from __future__ import annotations

import json
import os
import secrets
import stat
from contextlib import contextmanager

import psutil

from ml_stack import home
from ml_stack.files import writing

from .onboard.lan import require_local

ENV = 'ML_STACK_WSL_NETWORK'
LIMIT = 4096


def _path():
    return home.state('runtime', 'windows-network.json')


def _private(info, directory=False):
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not kind(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.geteuid():
        raise OSError('WSL discovery registration must be private and owned by this user')


def _read():
    path = _path()
    _private(path.parent.lstat(), directory=True)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        _private(os.fstat(fd))
        raw = os.read(fd, LIMIT + 1)
    finally:
        os.close(fd)
    if len(raw) > LIMIT:
        raise OSError('WSL discovery registration exceeds its message limit')
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise OSError('Invalid WSL discovery registration') from exc


def _validate(config):
    try:
        host, port = config['address']
        token = config['token']
        if not isinstance(host, str) or not isinstance(port, int) or isinstance(port, bool) or not 0 < port < 65536:
            raise ValueError('invalid address')
        require_local(host, port)
        if not isinstance(token, str) or len(token) != 64 or any(c not in '0123456789abcdef' for c in token):
            raise ValueError('invalid transport token')
    except (KeyError, TypeError, ValueError) as exc:
        raise OSError('Invalid WSL discovery bridge configuration') from exc
    return config


def _supported():
    return os.name == 'posix'


def configuration():
    """Return the explicit bridge or the current same-user daemon's registration."""
    if ENV in os.environ:
        try:
            if len(os.environ[ENV].encode("utf-8")) > LIMIT:
                raise ValueError('oversized configuration')
            return _validate(json.loads(os.environ[ENV]))
        except ValueError as exc:
            raise OSError('Invalid WSL discovery bridge configuration') from exc
    if not _supported() or not os.path.lexists(_path()):
        return None
    record = _read()
    try:
        process = psutil.Process(record['pid'])
        if process.create_time() != record['started'] or not process.is_running():
            raise ValueError('stale daemon')
        if process.uids().effective != os.geteuid():
            raise ValueError('foreign daemon')
        if 'ml_stack.cli.wsl_daemon' not in process.cmdline():
            raise ValueError('different daemon')
        return _validate(record['config'])
    except (KeyError, TypeError, ValueError, psutil.Error) as exc:
        raise OSError('WSL discovery daemon registration is stale or invalid; restart ml-stack from Windows') from exc


@contextmanager
def registered():
    """Publish the owned WSL daemon's bridge until it exits."""
    explicit = os.environ.get(ENV)
    if not explicit:
        yield
        return
    config = configuration()
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _private(path.parent.lstat(), directory=True)
    record = {'pid': os.getpid(), 'started': psutil.Process().create_time(),
              'nonce': secrets.token_hex(32), 'config': config}
    with writing(path) as temporary:
        temporary.chmod(0o600)
        temporary.write_text(json.dumps(record), encoding='utf-8')
    try:
        yield
    finally:
        try:
            if _read() == record:
                path.unlink()
        except FileNotFoundError:
            pass
