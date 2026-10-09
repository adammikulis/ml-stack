"""Authenticated agent interfaces operate on tasks without granting authority."""

import json

import pytest
from taskboard_kit import accepted, board, proposed
from workspace_kit import cli as run_cli

from ml_stack import mcp
from ml_stack.memory import vault
from ml_stack.workspace import task_outcomes, tools
from ml_stack.workspace.identity import TOKEN_ENV, Denied

__all__ = ['board']


def actor(monkeypatch, board, token):
    monkeypatch.setattr(tools, 'Workspace', lambda: board.ws)
    monkeypatch.setenv(TOKEN_ENV, token)


def test_cli_lists_and_reads_task_with_fixed_actor(board):
    result = run_cli(board.base, board.child, 'tasks', '--json')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['tasks'][0]['id'] == board.task['id']
    result = run_cli(board.base, board.child, 'task', board.task['id'], '--json')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['acceptance'] == ['Replay passes']
    result = run_cli(board.base, '', 'tasks', '--json')
    assert result.returncode == 3


def test_mcp_claim_checkpoint_submit_and_pending_review_share_actual_graph(board, monkeypatch):
    actor(monkeypatch, board, board.child)
    tools.workspace_task_claim(board.task['id'], board.allocation['allocation_id'])
    tools.workspace_task_heartbeat(board.task['id'])
    tools.workspace_task_checkpoint(board.task['id'], {'summary': 'Native replay collected'})
    tools.workspace_task_submit(board.task['id'], {'artifacts': {'replay.json': 'a' * 64},
                                                   'checks': [{'name': 'Worker check', 'passed': True}]})
    assert tools.workspace_task(board.task['id'])['checkpoints'][0]['summary'] == 'Native replay collected'
    actor(monkeypatch, board, board.parent)
    def locked(*args):
        raise vault.KeyUnavailable('isolated locked ledger')
    monkeypatch.setattr(task_outcomes.task_credit, 'verify_task', locked)
    result = tools.workspace_task_review(board.task['id'], accepted())
    assert result['outcome'] == 'accepted' and result['credit']['state'] == 'pending'
    assert tools.workspace_task(board.task['id'])['state'] == 'accepted'
    assert tools.workspace_task_credit(board.task['id'])['state'] == 'pending'


@pytest.mark.redteam
def test_mcp_cannot_self_review_or_claim_forged_resource(board, monkeypatch):
    actor(monkeypatch, board, board.child)
    with pytest.raises(Denied):
        tools.workspace_task_claim(board.task['id'], 'foreign-allocation')
    assert tools.workspace_task(board.task['id'])['state'] == 'queued'
    tools.workspace_task_claim(board.task['id'], board.allocation['allocation_id'])
    tools.workspace_task_submit(board.task['id'], {'artifacts': {'replay.json': 'a' * 64},
                                                   'checks': [{'name': 'Worker check', 'passed': True}]})
    with pytest.raises(Denied):
        tools.workspace_task_review(board.task['id'], accepted())
    assert tools.workspace_task(board.task['id'])['state'] == 'review'


def test_mcp_exports_native_task_schemas_without_actor_or_permission_fields():
    listed = {tool.public()['name']: tool.public() for tool in mcp.TOOLS}
    for name in tools.HINTS:
        if name.startswith('workspace_task'):
            assert 'token' not in listed[name]['inputSchema']['properties']
            assert 'actor' not in listed[name]['inputSchema']['properties']
    assert 'workspace_task_create' not in listed
    assert listed['workspace_tasks']['annotations']['readOnlyHint']
    assert not listed['workspace_task_review']['annotations']['readOnlyHint']


@pytest.mark.redteam
def test_native_cli_review_stdin_preserves_independent_authority(board):
    proposed(board)
    refused = run_cli(board.base, board.child, 'task-review', board.task['id'], '-',
                      '--json', input=json.dumps(accepted()))
    assert refused.returncode == 3
    assert board.board.get(board.parent, board.task['id'])['state'] == 'review'
    malformed = run_cli(board.base, board.parent, 'task-review', board.task['id'], '-',
                        '--json', input='[]')
    assert malformed.returncode != 0
    assert board.board.get(board.parent, board.task['id'])['review'] is None
    result = run_cli(board.base, board.parent, 'task-review', board.task['id'], '-',
                     '--json', input=json.dumps(accepted()))
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response['outcome'] == 'accepted'
    assert response['credit']['state'] == 'pending'
    assert board.board.get(board.parent, board.task['id'])['state'] == 'completed'
