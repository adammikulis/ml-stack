"""Owned launcher replacement admission over real loopback HTTP."""

import json
import threading
from http.server import BaseHTTPRequestHandler

import pytest

from ml_stack.fleet.daemon_control import (
    ROUTE,
    Control,
    ControlError,
    protected,
    request_replacement,
)
from ml_stack.http import Server, ServerError, request_json
from ml_stack.lock import Busy, only_one


@pytest.fixture
def device(tmp_path):
    stopped = threading.Event()
    busy = [False]
    active = threading.Event()
    release = threading.Event()
    holder = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if holder[0].route(self):
                return
            active.set()
            release.wait(5)
            self._send(200, {'accepted': True})

        def _body(self, limit):
            length = int(self.headers.get('Content-Length', '0'))
            if length > limit:
                self._send(413, {'error': 'too large'})
                return None
            return self.rfile.read(length)

        def _send(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    Handler.do_POST = protected(Handler.do_POST, lambda: holder[0])
    server = Server(('127.0.0.1', 0), Handler)
    port = server.server_address[1]
    control = Control(tmp_path, port, lambda: not busy[0],
                      lambda: only_one(tmp_path / 'runtime-install.lock', wait=False), stopped.set)
    holder.append(control)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield tmp_path, port, control, busy, stopped, active, release
    release.set()
    server.shutdown()
    server.server_close()
    control.close()


def test_owned_idle_replacement_retains_admission_and_rejects_new_work(device):
    root, port, control, _busy, stopped, active, _release = device
    answer = request_replacement(root, port, {'launcher_control': control.instance}, 'a' * 40)
    assert answer['stopping'] and stopped.wait(2)
    with pytest.raises(Busy), only_one(root / 'runtime-install.lock', wait=False):
        pass
    with pytest.raises(ServerError):
        request_json(f'http://127.0.0.1:{port}/jobs', method='POST', data=b'{}')
    assert not active.is_set()
    control.close()
    with only_one(root / 'runtime-install.lock', wait=False):
        assert not control.path.exists()


def test_busy_device_keeps_work_and_allows_later_retry(device):
    root, port, control, busy, stopped, _active, _release = device
    busy[0] = True
    with pytest.raises(ServerError):
        request_replacement(root, port, {'launcher_control': control.instance}, 'a' * 40)
    assert not stopped.is_set()
    busy[0] = False
    assert request_replacement(root, port, {'launcher_control': control.instance}, 'a' * 40)['stopping']


def test_request_in_progress_cannot_race_idle_replacement(device):
    root, port, control, _busy, stopped, active, release = device
    replies = []
    thread = threading.Thread(target=lambda: replies.append(request_json(
        f'http://127.0.0.1:{port}/jobs', method='POST', data=b'{}')))
    thread.start()
    assert active.wait(2)
    with pytest.raises(ServerError):
        request_replacement(root, port, {'launcher_control': control.instance}, 'a' * 40)
    assert not stopped.is_set()
    release.set()
    thread.join(3)
    assert replies == [{'accepted': True}]
    assert request_replacement(root, port, {'launcher_control': control.instance}, 'a' * 40)['stopping']


def test_other_control_instance_never_requests_shutdown(device):
    root, port, _control, _busy, stopped, _active, _release = device
    with pytest.raises(ControlError, match='does not match'):
        request_replacement(root, port, {'launcher_control': '0' * 32}, 'a' * 40)
    assert not stopped.is_set()


def test_cluster_or_unsigned_identity_cannot_replace_daemon(device):
    _root, port, control, _busy, stopped, _active, _release = device
    body = json.dumps({'instance': control.instance}).encode()
    for token in ('', 'mac:unrelated-cluster'):
        with pytest.raises(ServerError):
            request_json(f'http://127.0.0.1:{port}{ROUTE}', method='POST', data=body, token=token)
    assert not stopped.is_set()


def test_control_record_rejects_symlink_and_public_mode(device, tmp_path):
    root, port, control, _busy, stopped, _active, _release = device
    original = control.path.read_bytes()
    control.path.chmod(0o644)
    with pytest.raises(ControlError, match='account-private'):
        request_replacement(root, port, {'launcher_control': control.instance}, 'a' * 40)
    control.path.unlink()
    target = tmp_path / 'other-record'
    target.write_bytes(original)
    control.path.symlink_to(target)
    with pytest.raises(ControlError):
        request_replacement(root, port, {'launcher_control': control.instance}, 'a' * 40)
    assert not stopped.is_set()


def test_disconnected_response_still_starts_accepted_shutdown(device):
    from types import SimpleNamespace

    from ml_stack.macauth import sign

    _root, port, control, _busy, stopped, _active, _release = device
    body = json.dumps({'instance': control.instance}).encode()
    host = f'127.0.0.1:{port}'

    def disconnected(*_args):
        raise BrokenPipeError('client disconnected')

    handler = SimpleNamespace(path=ROUTE, client_address=('127.0.0.1', 1234),
                              headers={'Host': host, **sign(control.capability, 'POST', f'http://{host}{ROUTE}', body)},
                              _body=lambda _limit: body, _send=disconnected)
    with pytest.raises(BrokenPipeError):
        control.route(handler)
    assert stopped.wait(2)


@pytest.mark.parametrize('verb', ['GET', 'HEAD', 'PUT', 'DELETE'])
def test_each_protected_http_method_refuses_new_requests_during_drain(device, verb):
    from types import SimpleNamespace

    root, port, control, _busy, _stopped, _active, _release = device
    request_replacement(root, port, {'launcher_control': control.instance}, 'a' * 40)
    replies = []
    handler = SimpleNamespace(command=verb, path='/jobs', _send=lambda *args: replies.append(args))
    protected(lambda _handler: pytest.fail('admitted work after drain'), lambda: control)(handler)
    assert replies[0][0] == 503
    assert handler.close_connection


@pytest.mark.parametrize('failure', [ValueError, OSError])
def test_failed_idle_check_releases_admission_and_allows_retry(device, failure):
    root, port, control, _busy, stopped, _active, _release = device

    def unavailable():
        raise failure('idle state unavailable')

    control.idle = unavailable
    with pytest.raises(ServerError):
        request_replacement(root, port, {'launcher_control': control.instance}, 'a' * 40)
    assert not control.stopping and not stopped.is_set()
    with only_one(root / 'runtime-install.lock', wait=False):
        pass
    control.idle = lambda: True
    assert request_replacement(root, port, {'launcher_control': control.instance}, 'a' * 40)['stopping']


def test_valid_machine_control_signature_from_lan_is_refused_before_body(device):
    from types import SimpleNamespace

    from ml_stack.macauth import sign

    _root, port, control, _busy, stopped, _active, _release = device
    body = json.dumps({'instance': control.instance}).encode()
    host = f'127.0.0.1:{port}'
    replies = []
    handler = SimpleNamespace(path=ROUTE, client_address=('192.0.2.14', 1234),
                              headers={'Host': host, **sign(control.capability, 'POST', f'http://{host}{ROUTE}', body)},
                              _body=lambda _limit: pytest.fail('read remote control body'),
                              _send=lambda *args: replies.append(args))
    assert control.route(handler)
    assert replies[0][0] == 403
    assert not stopped.is_set() and not control.stopping


def test_runtime_control_preserves_queued_jobs_even_when_otherwise_idle(tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from ml_stack.fleet.daemon_control import create
    from ml_stack.macauth import sign

    stopped = threading.Event()
    queued = [1]
    runtime = SimpleNamespace(root=tmp_path, port=8770, nothing_running=lambda: True,
                              runner=SimpleNamespace(status=lambda: {'queued': queued[0]}),
                              update_admission=nullcontext, httpd=SimpleNamespace(shutdown=stopped.set))
    control = create(runtime)
    body = json.dumps({'instance': control.instance}).encode()
    host = '127.0.0.1:8770'
    replies = []
    handler = SimpleNamespace(path=ROUTE, client_address=('127.0.0.1', 1234),
                              headers={'Host': host, **sign(control.capability, 'POST', f'http://{host}{ROUTE}', body)},
                              _body=lambda _limit: body, _send=lambda *args: replies.append(args))
    try:
        assert control.route(handler)
        assert replies[0][0] == 409
        assert queued == [1] and not stopped.is_set() and not control.stopping
    finally:
        control.close()
