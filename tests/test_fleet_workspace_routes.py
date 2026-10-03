"""Workspace data and job routes against a daemon socket."""

from __future__ import annotations

import pytest
from test_fleet_ui import Serving


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
    code, result, _ = daemon.call('/ui/workspace/files?path=datasets')
    assert code == 200
    assert result['files'][0]['path'] == 'datasets/sample.jsonl'
    code, result, _ = daemon.call('/ui/workspace/file?path=datasets/sample.jsonl')
    assert result['text'] == '{"input":"stop"}\n'
    assert not result['truncated']
    assert daemon.call('/ui/workspace/file?path=../outside')[0] == 400
    outside = tmp_path / 'outside'
    outside.write_text('private')
    (daemon.files / 'escape').symlink_to(outside)
    assert daemon.call('/ui/workspace/file?path=escape')[0] == 400
    assert daemon.call('/ui/workspace/file', method='POST',
                       body={'path': 'datasets/sample.jsonl', 'text': 'replace'})[0] == 400


def test_workspace_requires_ui_header(daemon):
    assert daemon.call('/ui/workspace/files', ui_header=False)[0] == 403
    assert daemon.call('/ui/workspace/jobs', ui_header=False)[0] == 403
    assert daemon.call('/ui/gym/catalogue', ui_header=False)[0] == 403


def test_job_preview_is_not_submitted_and_commands_are_installed(daemon, monkeypatch):
    from ml_stack.fleet import workspace_routes
    monkeypatch.setattr(workspace_routes, 'commands', lambda: [
        {'name': 'ml-stack-doctor', 'entry': 'ml_stack.doctor:main'}])
    code, result, _ = daemon.call('/ui/workspace/jobs', method='POST', body={
        'command': 'ml-stack-doctor', 'args': ['--help'], 'preview': True})
    assert code == 200
    assert result['argv'] == ['ml-stack-doctor', '--help']
    assert daemon.runner.snapshot() == []
    assert daemon.call('/ui/workspace/jobs', method='POST', body={
        'command': 'python', 'args': []})[0] == 400
    assert daemon.call('/ui/workspace/jobs', method='POST', body={
        'command': 'ml-stack-doctor', 'args': ['--detach']})[0] == 400
    assert daemon.call('/ui/workspace/jobs', method='POST', body={
        'command': 'ml-stack-doctor', 'args': 'oops'})[0] == 400


def test_job_log_metrics_and_cancel(daemon):
    daemon.runner.gate = lambda: (False, 'test hold')
    job = daemon.runner.submit('check', ['unused'], str(daemon.files))
    daemon.runner.log_path(job.id).write_text('step 1\n')
    (daemon.runner.job_dir(job.id) / 'metrics.jsonl').write_text('{"loss":0.5}\npartial\n')
    code, result, _ = daemon.call('/ui/workspace/jobs/' + job.id)
    assert code == 200
    assert result['log'] == 'step 1\n'
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
    from ml_stack.decide_cli import COMMANDS
    parsed = COMMANDS.parser().parse_args(args)
    assert parsed.cmd == args[0]


def test_recipe_contracts_are_exposed(daemon):
    code, result, _ = daemon.call('/ui/workspace/recipes')
    assert code == 200
    recipes = {item['id']: item for item in result['recipes']}
    assert set(recipes) >= {'text-lm', 'classify-text', 'tool-calls'}
    assert recipes['text-lm']['fields'][0]['name'] == 'steps'
