"""Owned daemon replacement admission and launcher handoff."""

from __future__ import annotations

import contextlib
import ipaddress
import json
import os
import secrets
import stat
import threading
from functools import wraps
from pathlib import Path

from ml_stack import macauth, sealing
from ml_stack.files import writing
from ml_stack.fleet import updates
from ml_stack.http import Sealed, open_stream
from ml_stack.lock import Busy
from ml_stack.serve.reclaim import busy_now
from ml_stack.windows_private import restrict, validate

ROUTE = '/launcher/replace'
MAX_RECORD = 4096


class ControlError(RuntimeError):
    """The installed launcher cannot replace the running daemon."""


def _directory(root: Path) -> Path:
    root = root.absolute()
    path = root / 'launcher-control'
    if os.name == 'nt':
        validate(path)
    for candidate in (path, root, *root.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ControlError('Daemon launcher storage must be an owned plain directory.')
        if os.name != 'nt' and candidate in (path, root) and info.st_uid != os.getuid():
            raise ControlError('Daemon launcher storage belongs to another account.')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == 'nt':
        restrict(path)
    elif path.stat().st_mode & 0o077:
        raise ControlError('Daemon launcher storage must have owner-only permissions.')
    return path


def _record(root: Path, port: int) -> dict:
    path = _directory(root) / f'{port}.json'
    if os.name == 'nt':
        validate(path)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ControlError('This daemon has no owned launcher control; it must exit before replacement.') from exc
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RECORD:
            raise ControlError('Daemon launcher control record is invalid.')
        if os.name != 'nt' and (info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise ControlError('Daemon launcher control record must be account-private.')
        try:
            value = json.loads(stream.read(MAX_RECORD + 1))
        except (ValueError, UnicodeError) as exc:
            raise ControlError('Daemon launcher control record is invalid.') from exc
    if not isinstance(value, dict) or value.get('port') != port or value.get('version') != 1:
        raise ControlError('Daemon launcher control record is invalid.')
    return value


def request_replacement(root: Path, port: int, running: dict, expected: str) -> dict:
    """Ask the owned older daemon to stop at its maintained idle boundary."""
    record = _record(root, port)
    instance = running.get('launcher_control')
    capability = record.get('capability', '')
    if not isinstance(instance, str) or len(instance) != 32 or record.get('instance') != instance:
        raise ControlError('The running daemon does not match this launcher control record.')
    if not isinstance(capability, str) or not capability.startswith(macauth.PREFIX) or len(capability) > 128:
        raise ControlError('Daemon launcher control record is invalid.')
    endpoint = f'http://127.0.0.1:{port}{ROUTE}'

    def guard(url):
        if url != endpoint:
            raise ControlError('Daemon replacement cannot leave its owned loopback endpoint.')
        return url

    body = json.dumps({'version': 1, 'instance': instance, 'expected': expected}).encode()
    with open_stream(endpoint, method='POST', data=body, token=capability, timeout=10,
                     headers={'Content-Type': 'application/json', sealing.HEADER: '2'}, guard=guard) as response:
        if response.headers.get(sealing.HEADER) != '1':
            raise ControlError('Daemon replacement acknowledgment must authenticate with sealing.')
        if response.headers.get('Content-Type', '').split(';', 1)[0].strip().lower() != 'application/json':
            raise ControlError('Daemon replacement acknowledgment must be JSON.')
        limit = MAX_RECORD + sealing.NONCE_BYTES + 16
        wire = response.read(limit + 1)
        if len(wire) > limit:
            raise ControlError('Daemon replacement acknowledgment exceeds its size limit.')
        opened = getattr(response, 'sealed', Sealed()).open(response.status, response.headers, wire)
        if len(opened) > MAX_RECORD:
            raise ControlError('Daemon replacement acknowledgment exceeds its size limit.')
        try:
            answer = json.loads(opened)
        except (ValueError, UnicodeError) as exc:
            raise ControlError('Daemon replacement acknowledgment is not valid JSON.') from exc
    if not isinstance(answer, dict) or answer.get('instance') != instance or answer.get('stopping') is not True:
        raise ControlError('Daemon replacement was not acknowledged.')
    return answer


class Control:
    def __init__(self, root: Path, port: int, idle, admission, shutdown):
        self.path = _directory(root) / f'{port}.json'
        self.instance = secrets.token_hex(16)
        self.capability = macauth.PREFIX + secrets.token_urlsafe(32)
        self.auth = macauth.Authenticator(lambda: [self.capability])
        self.idle, self.admission, self.shutdown = idle, admission, shutdown
        self.lock = threading.Lock()
        self.held = contextlib.ExitStack()
        self.stopping = False
        self.active = 0
        with writing(self.path) as temporary:
            if os.name == 'nt':
                restrict(temporary)
            temporary.write_text(json.dumps({'version': 1, 'port': port, 'instance': self.instance,
                                             'capability': self.capability}), encoding='utf-8')

    def route(self, handler) -> bool:
        if handler.path != ROUTE:
            return False
        if not ipaddress.ip_address(handler.client_address[0]).is_loopback:
            handler._send(403, {'error': 'Daemon replacement requires this machine.'})
            return True
        body = handler._body(1024)
        if body is None:
            return True
        verdict = self.auth.check('POST', handler.path, handler.headers, body)
        if not verdict.ok:
            handler._send(403, {'error': 'Daemon replacement requires its owned launcher.'})
            return True
        try:
            mode = handler.headers.get(sealing.HEADER, '')
            if mode in ('1', '2'):
                key = sealing.box_key(verdict.secret)
                host, target = macauth.parts(f"//{handler.headers.get('Host', '')}{handler.path}")
                handler._opening = (key, verdict, mode == '2', handler.headers)
                body = sealing.open_(key, body, sealing.request_data(
                    'POST', target, host, verdict.at, verdict.nonce))
            request = json.loads(body)
            if not isinstance(request, dict) or request.get('instance') != self.instance:
                raise ControlError('Daemon replacement instance does not match.')
            with self.lock:
                if self.stopping or self.active:
                    raise ControlError('Daemon has requests in progress; retry when it is idle.')
                with contextlib.ExitStack() as admitted:
                    admitted.enter_context(self.admission())
                    if not self.idle():
                        raise ControlError('Daemon has active work, downloads or setup; retry when it is idle.')
                    self.held = admitted.pop_all()
                    self.stopping = True
        except (ControlError, Busy, ValueError, OSError, sealing.SealError) as exc:
            handler._send(409, {'error': str(exc)})
            return True
        try:
            handler._send(202, {'instance': self.instance, 'stopping': True})
            handler.wfile.flush()
        finally:
            threading.Thread(target=self.shutdown, name='launcher-daemon-stop', daemon=True).start()
        return True

    def close(self) -> None:
        with self.lock:
            self.held.close()
            try:
                row = _record(self.path.parent.parent, int(self.path.stem))
            except ControlError:
                return
            if row.get('instance') == self.instance:
                self.path.unlink(missing_ok=True)


def protected(method, control):
    """Admit a complete HTTP operation before daemon replacement drains requests."""
    @wraps(method)
    def handle(handler):
        owner = control() if control else None
        if owner is None or (handler.command == 'POST' and handler.path == ROUTE):
            return method(handler)
        with owner.lock:
            if owner.stopping:
                handler.close_connection = True
                handler._send(503, {'error': 'Daemon replacement is in progress; retry shortly.'})
                return None
            owner.active += 1
        try:
            return method(handler)
        finally:
            with owner.lock:
                owner.active -= 1
    return handle


def create(runtime):
    """Publish launcher control for a configured daemon runtime."""
    def models_busy():
        try:
            with runtime.serving.path.open('rb') as stream:
                raw = stream.read(1024 * 1024 + 1)
        except FileNotFoundError:
            return False
        if len(raw) > 1024 * 1024:
            return True
        rows = json.loads(raw)
        if not isinstance(rows, list):
            return True
        served = runtime.serving.all()
        if len(rows) != len(served):
            return True
        for row, server in zip(rows, served, strict=True):
            if (not isinstance(row, dict) or type(row.get('port')) is not int
                    or not 1 <= row['port'] <= 65535 or row['port'] != server.port):
                return True
        return any(busy_now(f'http://127.0.0.1:{server.port}') is not False for server in served)

    idle = updates.quiet(
        jobs=lambda: runtime.background_busy() or bool(runtime.runner.status()['queued']),
        measuring=lambda: bool(runtime.bench_host.measuring()),
        leases=models_busy,
    )
    return Control(runtime.root, runtime.port, idle,
                   runtime.update_admission, lambda: runtime.httpd.shutdown())
