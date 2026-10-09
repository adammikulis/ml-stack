"""Workspace data and job routes against a daemon socket."""

from __future__ import annotations

import os
import subprocess

import pytest
from test_fleet_ui import Serving

from poolhouse.fleet.workspace_routes import PURPOSES


@pytest.fixture
def daemon(tmp_path):
    instance = Serving(tmp_path)
    try:
        yield instance
    finally:
        instance.close()


def test_data_upload_preview_and_path_boundary(daemon, tmp_path):
    code, result, _ = daemon.call('/ui/workspace/file', method='POST',
                                  body={'path': 'datasets/sample.jsonl', 'text': '{"input":"stop"}\n'})
    assert code == 201
    assert result['path'] == 'datasets/sample.jsonl'
    assert (daemon.files / 'datasets' / 'sample.jsonl').read_bytes() == b'{"input":"stop"}\n'
    code, result, _ = daemon.call('/ui/workspace/files?path=datasets')
    assert code == 200
    assert result['files'][0]['path'] == 'datasets/sample.jsonl'
    code, result, _ = daemon.call('/ui/workspace/file?path=datasets/sample.jsonl')
    assert result['text'] == '{"input":"stop"}\n'
    assert not result['truncated']
    assert daemon.call('/ui/workspace/file?path=../outside')[0] == 400
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'private.txt').write_text('private')
    alias = daemon.files / 'escape'
    if os.name == 'nt':
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(alias), str(outside)],
                       capture_output=True, check=True)
    else:
        alias.symlink_to(outside, target_is_directory=True)
    try:
        assert daemon.call('/ui/workspace/file?path=escape/private.txt')[0] == 400
    finally:
        alias.rmdir() if os.name == 'nt' else alias.unlink()
    assert daemon.call('/ui/workspace/file', method='POST',
                       body={'path': 'datasets/sample.jsonl', 'text': 'replace'})[0] == 400


def test_workspace_requires_ui_header(daemon):
    assert daemon.call('/ui/workspace/files', ui_header=False)[0] == 403
    assert daemon.call('/ui/workspace/jobs', ui_header=False)[0] == 403
    assert daemon.call('/ui/gym/catalogue', ui_header=False)[0] == 403


def test_job_preview_is_not_submitted_and_commands_are_installed(daemon, monkeypatch):
    from poolhouse.fleet import workspace_routes
    monkeypatch.setattr(workspace_routes, 'commands', lambda: [
        {'name': 'poolhouse-doctor', 'entry': 'poolhouse.doctor:main'}])
    code, result, _ = daemon.call('/ui/workspace/jobs', method='POST', body={
        'command': 'poolhouse-doctor', 'args': ['--help'], 'preview': True})
    assert code == 200
    assert result['argv'] == ['poolhouse-doctor', '--help']
    assert daemon.runner.snapshot() == []
    assert daemon.call('/ui/workspace/jobs', method='POST', body={
        'command': 'python', 'args': []})[0] == 400
    assert daemon.call('/ui/workspace/jobs', method='POST', body={
        'command': 'poolhouse-doctor', 'args': ['--detach']})[0] == 400
    assert daemon.call('/ui/workspace/jobs', method='POST', body={
        'command': 'poolhouse-doctor', 'args': 'oops'})[0] == 400


def test_job_log_metrics_and_cancel(daemon):
    daemon.runner.gate = lambda: (False, 'test hold')
    job = daemon.runner.submit('check', ['unused'], str(daemon.files))
    daemon.runner.log_path(job.id).write_bytes(b'step 1\r\nstep 2\nprogress\rnext')
    (daemon.runner.job_dir(job.id) / 'metrics.jsonl').write_text('{"loss":0.5}\npartial\n')
    code, result, _ = daemon.call('/ui/workspace/jobs/' + job.id)
    assert code == 200
    assert result['log'] == 'step 1\nstep 2\nprogress\rnext'
    assert result['metrics'] == [{'loss': 0.5}]
    code, result, _ = daemon.call('/ui/workspace/jobs/' + job.id + '/stop', method='POST', body={})
    assert code == 200
    assert result['state'] == 'stopped'


@pytest.mark.parametrize('args', [
    ['ask', 'Choose action', '--state', 'Car ahead', '--option', 'stop', '--option', 'go',
     '--json', '--backend', 'pointer'],
    ['eval', 'guards', '--out', 'decisions/report.json', '--backend', 'pointer'],
    ['calibrate', 'guards', '--out', 'decisions/report.json', '--backend', 'pointer'],
    ['bench', 'guards', '--out', 'decisions/report.json', '--backend', 'pointer'],
    ['list'], ['fetch'],
    ['train', '--data', 'cases.jsonl', '--name', 'demo', '--out', 'models/demo',
     '--steps', '60', '--batch-size', '4', '--lr', '.0002', '--rank', '16', '--seed', '0', '--dry-run'],
])
def test_decision_workflow_arguments_match_command_contract(args):
    from poolhouse.decide_cli import COMMANDS
    parsed = COMMANDS.parser().parse_args(args)
    assert parsed.cmd == args[0]


def test_recipe_contracts_are_exposed(daemon):
    code, result, _ = daemon.call('/ui/workspace/recipes')
    assert code == 200
    recipes = {item['id']: item for item in result['recipes']}
    assert set(recipes) >= {'text-lm', 'classify-text', 'tool-calls'}
    assert recipes['text-lm']['fields'][0]['name'] == 'steps'


def test_frozen_gym_uses_installed_environment(daemon, monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    configured = []
    from poolhouse.fleet import gym_interpreters, gym_routes
    monkeypatch.delenv('POOLHOUSE_GYM_PYTHON', raising=False)
    monkeypatch.setenv('POOLHOUSE_GYM_PYTHONS', '{}')
    monkeypatch.setattr(gym_routes, 'catalogue', lambda: [{'id': 'car', 'available': True}])
    monkeypatch.setattr(gym_interpreters, 'manager', SimpleNamespace(
        configure=lambda python: configured.append(python), configure_map=lambda mapping: None))
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    python = tmp_path / 'env' / 'bin' / 'python'
    daemon.ui.environment = SimpleNamespace(exists=True, python=python)
    code, result, _ = daemon.call('/ui/gym/catalogue')
    assert code == 200
    assert result['environments'][0]['id'] == 'car'
    assert configured == [python]


def test_recording_review_boundary_and_export_handoff(daemon, tmp_path, monkeypatch):
    import json

    root = tmp_path / 'gym'
    recording = root / 'session-a'
    recording.mkdir(parents=True)
    snapshot = {'id': 'session-a', 'sequence': 2, 'episode_id': 1, 'environment': 'warehouse',
                'actions': ['stop', 'forward'], 'action': 0, 'reward': 1., 'terminated': False,
                'truncated': False, 'observation': [0., 1.], 'frame': None,
                'transition': {'episode_id': 1, 'sequence': 2, 'observation': [1., 0.]}}
    (recording / 'trajectory.jsonl').write_text(json.dumps(snapshot) + '\n')
    from poolhouse.fleet import gym_recording_routes
    monkeypatch.setattr(gym_recording_routes, 'artifact_root', lambda: root)

    def export(trajectory, reviews, output):
        assert trajectory == recording / 'trajectory.jsonl'
        assert json.loads(reviews.read_text())['label'] == 'stop'
        output.write_text('{"reviewed":true}\n')
        return 1

    monkeypatch.setattr(gym_recording_routes, 'export_reviewed', export)
    code, result, _ = daemon.call('/ui/gym/recordings')
    assert code == 200 and result['recordings'][0]['id'] == 'session-a'
    code, result, _ = daemon.call('/ui/gym/recordings/session-a')
    assert code == 200 and result['steps'][0]['sequence'] == 2
    code, result, _ = daemon.call('/ui/gym/recordings/session-a/frame?sequence=2')
    assert code == 200 and result['transition']['observation'] == [1., 0.]
    assert daemon.call('/ui/gym/recordings/../outside')[0] == 400
    body = {'episode_id': 1, 'sequence': 2, 'label': 'invented'}
    assert daemon.call('/ui/gym/recordings/session-a/review', method='POST', body=body)[0] == 400
    body['label'] = 'stop'
    assert daemon.call('/ui/gym/recordings/session-a/review', method='POST', body=body)[0] == 200
    code, result, _ = daemon.call('/ui/gym/recordings/session-a/export', method='POST',
                                  body={'path': 'datasets/reviewed.jsonl'})
    assert code == 201 and result['cases'] == 1
    assert (daemon.files / result['path']).is_file()


def test_specialist_descriptions_cover_native_chat_memory_and_gym():
    assert 'approve' in PURPOSES['chat']
    assert 'export' in PURPOSES['memory']
    assert 'evaluate' in PURPOSES['gym']
    assert 'do' not in PURPOSES


def test_gym_jobs_use_saved_example_python_without_shell_expansion(daemon, monkeypatch, tmp_path):
    from poolhouse.fleet import workspace_routes
    from poolhouse.fleet.settings import Settings

    monkeypatch.setattr(workspace_routes, 'commands', lambda: [{'name': 'poolhouse-gym'}])
    monkeypatch.setenv('POOLHOUSE_GYM_PYTHONS', '{}')
    python = tmp_path / 'Python with spaces' / 'bin' / 'python'
    python.parent.mkdir(parents=True)
    python.write_text('')
    daemon.ui.settings.gym_pythons = {'drone': str(python)}
    daemon.ui.settings.save(daemon.ui.settings_path)
    daemon.ui.settings = Settings.load(daemon.ui.settings_path)
    args = ['train', 'drone', '--config', '{"note":"$(touch /tmp/nope); a b"}']
    code, result, _ = daemon.call('/ui/workspace/jobs', method='POST', body={
        'command': 'poolhouse-gym', 'args': args, 'preview': True})
    assert code == 200
    assert result['argv'] == [str(python), '-m', 'poolhouse.gym.cli', *args]
    assert daemon.runner.snapshot() == []
    assert daemon.call('/ui/workspace/jobs', method='POST', body={
        'command': 'poolhouse-gym', 'args': ['train', '../../escape']})[0] == 400
    monkeypatch.setenv('POOLHOUSE_GYM_PYTHONS', '{}')


@pytest.mark.slow
def test_saved_drone_interpreter_runs_a_queued_native_session(daemon, monkeypatch):
    import json
    import os
    import time
    from pathlib import Path

    from poolhouse.fleet import workspace_routes
    from poolhouse.fleet.settings import Settings

    python = os.environ.get('POOLHOUSE_TEST_DRONE_PYTHON')
    if not python or not Path(python).is_file():
        pytest.skip('Set POOLHOUSE_TEST_DRONE_PYTHON to the installed native drone Python')
    monkeypatch.setenv('POOLHOUSE_GYM_PYTHONS', '{}')
    monkeypatch.setattr(workspace_routes, 'commands', lambda: [{'name': 'poolhouse-gym'}])
    daemon.ui.settings.gym_pythons = {'drone': python}
    daemon.ui.settings.save(daemon.ui.settings_path)
    daemon.ui.settings = Settings.load(daemon.ui.settings_path)
    monkeypatch.setenv('PYTHONPATH', str(Path(__file__).resolve().parents[1] / 'src'))
    config = {'world': {'trees': 0, 'hikers': 1, 'fires': 0, 'n_agents': 1}}
    code, result, _ = daemon.call('/ui/workspace/jobs', method='POST', body={
        'command': 'poolhouse-gym', 'args': ['run', 'drone', '--steps', '1', '--action', '0',
                                         '--config', json.dumps(config)]})
    assert code == 202
    job = daemon.runner.jobs[result['id']]
    until = time.monotonic() + 45
    while job.state not in {'done', 'failed', 'stopped'} and time.monotonic() < until:
        time.sleep(.05)
    assert job.state == 'done', daemon.runner.log_path(job.id).read_text()
    assert job.argv[:3] == [python, '-m', 'poolhouse.gym.cli']
    output = daemon.runner.log_path(job.id).read_text()
    assert 'drone' in output and 'observation' in output
