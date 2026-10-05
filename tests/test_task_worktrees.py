"""Canonical task branches exclude prior uncommitted worker changes."""

import pytest
from workspace_kit import Kit

from ml_stack.net import git
from ml_stack.workspace import localagent, task_worktrees
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.taskboard import TaskBoard


def test_clean_task_baseline_claims_and_parent_authority(tmp_path):
    repository, source = tmp_path / 'main', tmp_path / 'source'
    repository.mkdir()
    git.run(['init', str(repository)])
    (repository / 'code.py').write_text('ORIGINAL = True\n')
    git.run(['add', 'code.py'], cwd=repository)
    git.run(['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
             'commit', '-m', 'baseline'], cwd=repository)
    git.run(['worktree', 'add', '-b', 'source', str(source)], cwd=repository)
    (source / 'code.py').write_text('CONTAMINATED = True\n')
    kit = Kit(tmp_path / 'workspace')
    parent = kit.agent('parent')
    worker = kit.ws.delegate(parent, 'worker')['id']
    localagent.save(kit.ws, localagent.Agent('local-worker', 'qwen', identity=worker,
                                           profile='coding', project=str(source)))
    task = TaskBoard(kit.ws).create(kit.owner, {'title': 'Person task', 'description': 'Fix current code',
                                             'acceptance': ['Checks pass']})['id']
    with pytest.raises(Denied, match='registered worker parent'):
        task_worktrees.prepare(kit.ws, kit.agent('foreign'), worker, task)
    prepared = task_worktrees.prepare(kit.ws, parent, worker, task)
    target = tmp_path / ('task-' + task.split(':')[1])
    assert prepared['project'] == str(target)
    assert (target / 'code.py').read_text() == 'ORIGINAL = True\n'
    assert (source / 'code.py').read_text() == 'CONTAMINATED = True\n'
    assert prepared['baseline_commit'] == git.head(source)
    assert task_worktrees.prepare(kit.ws, parent, worker, task) == prepared
    assert kit.ws.claims.who('worktree', str(target))['owner'] == 'parent'
