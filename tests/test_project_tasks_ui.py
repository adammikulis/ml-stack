"""Human-authorized task controls use the selected project store."""

import pytest
import test_project_board_ui as board_ui

from ml_stack.workspace import tokens
from ml_stack.workspace.taskboard import TaskBoard

project_board = board_ui.project_board
PROJECT = board_ui.PROJECT
OTHER = board_ui.OTHER


def task_call(project_board, project=PROJECT, **options):
    call, _, projects = project_board
    projects[project].name = 'Experiment' if project == PROJECT else 'Other experiment'
    return call('', project=project, path=f'/ui/projects/{project}/tasks', **options)


def test_tasks_preserve_records_and_ignore_unrelated_default_coordinator(project_board, monkeypatch):
    from ml_stack.workspace import coordinator_config

    monkeypatch.setattr(coordinator_config, 'load', lambda _: {'mode': 'remote'})
    _, workspaces, _ = project_board
    ws = workspaces[PROJECT]
    owner = tokens.read_file(tokens.directory(ws.base) / tokens.OWNER_FILE)
    existing = TaskBoard(ws).create(owner, {'title': 'Existing task', 'acceptance': ['Reviewed']})
    status, created = task_call(project_board, method='POST', body={'spec': {
        'title': 'New task', 'acceptance': ['Passed'], 'deps': [existing['id']]}})
    assert status == 200, created
    assert created['project'] == {'key': PROJECT, 'name': 'Experiment'}
    status, result = task_call(project_board)
    assert status == 200, result
    assert {task['id'] for task in result['tasks']} == {existing['id'], created['id']}
    assert result['overview']['remaining'] == 2 and result['overview']['queued'] == 2
    assert result['overview']['eta_seconds'] is None
    status, other = task_call(project_board, OTHER)
    assert status == 200 and other['tasks'] == []
    assert workspaces[OTHER].base != ws.base


@pytest.mark.parametrize('options', [{'cookie': 'forged'}, {'ip': '198.51.100.4'},
                                    {'method': 'POST', 'origin': 'https://outside.invalid',
                                     'body': {'spec': {'title': 'Forbidden', 'acceptance': ['No']}}}])
def test_tasks_reject_forged_remote_and_cross_origin_people(project_board, options):
    status, _ = task_call(project_board, **options)
    assert status == 403


def test_tasks_refuse_foreign_project_metadata_and_agent_owner(project_board):
    status, _ = task_call(project_board, method='POST', body={'spec': {
        'title': 'Wrong project', 'acceptance': ['No'], 'project': {'key': OTHER}}})
    assert status == 403
    _, workspaces, projects = project_board
    ws = workspaces[PROJECT]
    owner = tokens.read_file(tokens.directory(ws.base) / tokens.OWNER_FILE)
    tokens.store(ws.base, tokens.OWNER_FILE, ws.mint(owner, 'another-worker'))
    assert task_call(project_board)[0] == 403
    projects[OTHER].board_host = 'http://foreign:8770'
    assert task_call(project_board, OTHER)[0] == 409
