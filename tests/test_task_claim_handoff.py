"""Native task edits require an explicit parent assignment and a live resource lease."""

from dataclasses import replace
from pathlib import Path

import pytest
from taskboard_kit import board as _board_fixture

from ml_stack import harness_claims, harnesshook
from ml_stack.net import git
from ml_stack.workspace import localagent, resource_allocations, task_worktrees
from ml_stack.workspace.project import describe

board = _board_fixture
pytestmark = pytest.mark.redteam


@pytest.fixture
def native_task(board, tmp_path, monkeypatch):
    repository, source = tmp_path / 'repository', tmp_path / 'source'
    repository.mkdir()
    git.run(['init', str(repository)])
    git.run(['remote', 'add', 'origin', 'https://example.invalid/fixture.git'], cwd=repository)
    (repository / 'code.py').write_text('ORIGINAL = True\n')
    git.run(['add', 'code.py'], cwd=repository)
    git.run(['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'baseline'], cwd=repository)
    git.run(['worktree', 'add', '-b', 'source', str(source)], cwd=repository)
    runner = localagent.load(board.ws, 'native-worker')
    localagent.save(board.ws, replace(runner, project=str(source)))
    board.task = board.board.create(board.parent, {**board.spec, 'source_key': 'native-handoff'})
    board.scope = task_worktrees.prepare(board.ws, board.parent, board.worker_id, board.task['id'])
    board.project = Path(board.scope['project'])
    board.ws.registry.set_project(board.ws.auth(board.owner), 'lead', describe(str(source)))
    board.allocation = resource_allocations.assign(board.ws, board.parent, board.worker_id,
                                                   board.task['id'], 'native-grant')
    board.board.claim(board.child, board.task['id'], board.allocation['allocation_id'])
    monkeypatch.setattr(harness_claims, 'Workspace', lambda: board.ws)
    return board


def edit(board, actor):
    target = board.project / 'code.py'
    payload = {'tool_name': 'Edit', 'tool_input': {'file_path': str(target),
               'old_string': 'ORIGINAL', 'new_string': 'UPDATED'}, 'cwd': str(board.project)}
    result = harnesshook.pre(payload, harnesshook.Rail('plan-and-go', actor,
                                                     roots=[str(board.project)], wait_s=0))
    decision = result['hookSpecificOutput']['permissionDecision']
    if decision == 'allow':
        target.write_text(target.read_text().replace('ORIGINAL', 'UPDATED'))
    return decision


def test_explicit_parent_claim_handoff_allows_only_the_assigned_native_worker(native_task):
    board = native_task
    assert board.ws.claims.who('worktree', str(board.project))['owner'] == 'lead'
    other = board.ws.delegate(board.parent, 'unrelated')['id']
    assert edit(board, other) == 'deny'
    assert (board.project / 'code.py').read_text() == 'ORIGINAL = True\n'
    assert edit(board, board.worker_id) == 'allow'
    owner = next(row for row in board.ws.claims.listing(kind='worktree') if row['key'] == str(board.project))
    assert owner['owner'] == board.worker_id and owner['delegated_by'] == 'lead'
    assert owner['assignment'] == board.scope['id']
    assert (board.project / 'code.py').read_text() == 'UPDATED = True\n'
    board.now[0] += 121
    assert edit(board, board.worker_id) == 'deny'


def test_released_allocation_and_unrelated_parent_file_claim_remain_refused(native_task):
    board = native_task
    board.ws.claim(board.parent, 'file', str(board.project / 'code.py'))
    assert edit(board, board.worker_id) == 'deny'
    board.ws.release(board.parent, 'file', str(board.project / 'code.py'))
    board.status['servers'][0]['holders'] = []
    assert edit(board, board.worker_id) == 'deny'
    assert board.ws.claims.who('worktree', str(board.project))['owner'] == 'lead'
    assert (board.project / 'code.py').read_text() == 'ORIGINAL = True\n'
