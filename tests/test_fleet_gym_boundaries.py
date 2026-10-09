"""Simulation and workspace request boundaries over a daemon socket."""

import http.client
import json
from types import SimpleNamespace

import pytest
from test_fleet_ui import Serving

from ml_stack.fleet import gym_recording_routes, gym_routes, workspace_routes


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_GYM_FILES_ROOT", str(tmp_path / "untrusted-root"))
    server = Serving(tmp_path)
    try:
        yield server
    finally:
        server.close()


def raw_request(server, path, body):
    connection = http.client.HTTPConnection('127.0.0.1', server.port, timeout=5)
    try:
        connection.request('POST', path, body=body, headers={'X-ML-Stack-UI': '1'})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


@pytest.mark.parametrize('path', ['/ui/gym/sessions', '/ui/workspace/jobs', '/ui/workspace/file'])
@pytest.mark.parametrize('body', [b'null', b'[]', b'42', b'{', b'\xff'])
def test_malformed_json_never_reaches_worker_or_job(daemon, monkeypatch, path, body):
    calls = []
    monkeypatch.setattr(gym_routes, 'manager', SimpleNamespace(create=lambda *a, **kw: calls.append(kw)))
    assert raw_request(daemon, path, body)[0] == 400
    assert calls == []
    assert daemon.runner.snapshot() == []


@pytest.mark.parametrize('body', [
    {'environment': ['car']}, {'environment': 'car', 'config': []},
    {'environment': 'car', 'config': {'world': []}}, {'environment': 'car', 'seed': True},
    {'environment': 'car', 'seed': '2'}, {'environment': 'car', 'controller': []},
    {'environment': 'car', 'config': {'world': {'mode': 'manual', 'map_file': '../secret'}}},
])
def test_session_creation_rejects_structural_coercions(daemon, monkeypatch, body):
    calls = []
    monkeypatch.setattr(gym_routes, 'manager', SimpleNamespace(create=lambda *a, **kw: calls.append(kw)))
    assert daemon.call('/ui/gym/sessions', method='POST', body=body)[0] == 400
    assert calls == []


def test_unauthenticated_simulation_controls_do_not_dispatch(daemon, monkeypatch):
    calls = []
    monkeypatch.setattr(gym_routes, 'manager', SimpleNamespace(control=lambda *a, **kw: calls.append(kw)))
    assert daemon.call('/ui/gym/sessions/session', method='POST',
                       body={'command': 'play'}, ui_header=False)[0] == 403
    assert daemon.call('/ui/gym/sessions/session', method='POST',
                       body={'command': 'play', 'payload': []})[0] == 400
    assert calls == []
    assert daemon.call('/ui/gym/sessions', method='DELETE')[0] == 405


def test_manual_world_uses_daemon_root_and_rejects_symlink(daemon, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gym_routes, 'manager', SimpleNamespace(create=lambda *a, **kw: calls.append((a, kw)) or {'id': 'test'}))
    outside = tmp_path / 'private.json'
    outside.write_text('{}')
    (daemon.files / 'escape.json').symlink_to(outside)
    body = {'environment': 'car', 'config': {'world': {'mode': 'manual', 'map_file': 'escape.json'}}}
    assert daemon.call('/ui/gym/sessions', method='POST', body=body)[0] == 400
    assert calls == []
    (daemon.files / 'map.json').write_text('{}')
    body['config']['world']['map_file'] = 'map.json'
    assert daemon.call('/ui/gym/sessions', method='POST', body=body)[0] == 201
    assert calls[0][0] == ('car',)
    import os
    assert os.environ['ML_STACK_GYM_FILES_ROOT'] == str(daemon.files)


@pytest.mark.parametrize('extra', [{'metadata': []}, {'preview': 'yes'}, {'name': []}, {'args': ['--help', 1]}])
def test_job_structure_is_validated_before_submission(daemon, monkeypatch, extra):
    monkeypatch.setattr(workspace_routes, 'commands', lambda: [{'name': 'ml-stack-doctor'}])
    assert daemon.call('/ui/workspace/jobs', method='POST', body={'command': 'ml-stack-doctor', **extra})[0] == 400
    assert daemon.runner.snapshot() == []


def test_recording_frame_and_review_symlinks_remain_jailed(daemon, monkeypatch, tmp_path):
    root = tmp_path / 'gym'
    session = root / 'session'
    session.mkdir(parents=True)
    outside = tmp_path / 'secret'
    outside.write_text('private')
    row = {'sequence': 1, 'frame_path': str(outside), 'transition': {'episode_id': 0, 'sequence': 1}, 'actions': ['stop']}
    (session / 'trajectory.jsonl').write_text(json.dumps(row) + '\n')
    monkeypatch.setattr(gym_recording_routes, 'artifact_root', lambda: root)
    code, body, _ = daemon.call('/ui/gym/recordings/session/frame?sequence=1')
    assert code == 400 and 'private' not in json.dumps(body)
    assert daemon.call('/ui/gym/recordings/session/review', method='POST',
                       body={'episode_id': 0, 'sequence': 1, 'label': 'go'})[0] == 400
    (session / 'reviews.jsonl').symlink_to(outside)
    assert daemon.call('/ui/gym/recordings/session/review', method='POST',
                       body={'episode_id': 0, 'sequence': 1, 'label': 'stop'})[0] == 400
    assert outside.read_text() == 'private'


def test_native_manager_rejects_unknown_commands_without_dispatch(daemon, monkeypatch):
    from ml_stack.gym.runtime import SessionManager
    monkeypatch.setattr(gym_routes, 'manager', SessionManager())
    code, _, _ = daemon.call('/ui/gym/sessions/nonexistent', method='POST',
                            body={'command': 'shell', 'payload': {'args': ['touch', 'outside']}})
    assert code == 400


def test_workspace_argv_is_literal_and_never_shell_parsed(daemon, monkeypatch):
    monkeypatch.setattr(workspace_routes, 'commands', lambda: [{'name': 'ml-stack-doctor'}])
    value = '; touch outside $(whoami)'
    code, result, _ = daemon.call('/ui/workspace/jobs', method='POST',
        body={'command': 'ml-stack-doctor', 'args': [value], 'preview': True})
    assert code == 200 and result['argv'] == ['ml-stack-doctor', value]
    assert daemon.runner.snapshot() == []
