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
    assert not target.exists()
    assert prepared['state'] == 'reserved'
    from ml_stack.graph.store import GraphStore
    with GraphStore(kit.ws.base / 'coordination.db') as graph:
        task_worktrees.activate(graph, worker, task)
    assert (target / 'code.py').read_text() == 'ORIGINAL = True\n'
    assert (source / 'code.py').read_text() == 'CONTAMINATED = True\n'
    assert prepared['baseline_commit'] == git.head(source)
    assert task_worktrees.prepare(kit.ws, parent, worker, task) == {**prepared, 'state': 'active'}
    assert kit.ws.claims.who('worktree', str(target))['owner'] == 'parent'


def test_task_target_inside_a_checkout_is_refused_before_creation(tmp_path):
    from ml_stack import worktreerules
    primary = tmp_path / 'repo'
    primary.mkdir()
    git.run(['init', str(primary)])
    assert 'inside checkout' in worktreerules.worktree_refusal(primary / '.worktrees' / 'task', primary)
    assert not worktreerules.worktree_refusal(tmp_path / 'task', primary)


def test_shell_guard_refuses_nested_creation_in_primary_and_worktree(tmp_path):
    from ml_stack import worktreerules
    primary = tmp_path / 'repo'
    (primary / 'scripts' / 'hooks').mkdir(parents=True)
    (primary / 'scripts' / 'hooks' / 'primary-only').write_text('')
    (primary / 'code.py').write_text('ready = True\n')
    git.run(['init', str(primary)])
    git.run(['add', '.'], cwd=primary)
    git.run(['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'baseline'], cwd=primary)
    work = tmp_path / 'work'
    git.run(['worktree', 'add', '-b', 'work', str(work)], cwd=primary)
    assert 'inside checkout' in worktreerules.bash_refusal('git worktree add -b task .worktrees/task HEAD', str(primary))
    assert 'inside checkout' in worktreerules.bash_refusal('git worktree add -b task nested/task HEAD', str(work))
    assert not worktreerules.bash_refusal('git worktree add -b task ../task HEAD', str(primary))


def test_primary_baseline_is_read_only_and_execution_materializes_beside_it(tmp_path):
    from ml_stack.graph.store import GraphStore
    primary = tmp_path / 'repo'
    primary.mkdir()
    git.run(['init', str(primary)])
    (primary / 'code.py').write_text('ORIGINAL = True\n')
    git.run(['add', 'code.py'], cwd=primary)
    git.run(['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'baseline'], cwd=primary)
    kit = Kit(tmp_path / 'workspace')
    parent = kit.agent('parent')
    worker = kit.ws.delegate(parent, 'worker')['id']
    localagent.save(kit.ws, localagent.Agent('local-worker', 'qwen', identity=worker,
                                           profile='coding', project=str(primary)))
    task = TaskBoard(kit.ws).create(kit.owner, {'title': 'Task', 'description': 'Current code',
                                             'acceptance': ['Checks pass']})['id']
    reserved = task_worktrees.prepare(kit.ws, parent, worker, task)
    target = tmp_path / ('task-' + task.split(':')[1])
    assert reserved['project'] == str(target) and not target.exists()
    with GraphStore(kit.ws.base / 'coordination.db') as graph:
        task_worktrees.activate(graph, worker, task)
    assert (target / '.git').is_file()
    (target / 'code.py').write_text('CHANGED = True\n')
    assert (primary / 'code.py').read_text() == 'ORIGINAL = True\n'
    assert not git.run(['status', '--porcelain'], cwd=primary).stdout.strip()


def test_crash_created_checkout_is_validated_and_promoted_without_losing_changes(tmp_path):
    from ml_stack.graph.store import GraphStore
    primary = tmp_path / 'repo'
    primary.mkdir()
    git.run(['init', str(primary)])
    (primary / 'code.py').write_text('ORIGINAL = True\n')
    git.run(['add', 'code.py'], cwd=primary)
    git.run(['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'baseline'], cwd=primary)
    kit = Kit(tmp_path / 'workspace')
    parent = kit.agent('parent')
    worker = kit.ws.delegate(parent, 'worker')['id']
    localagent.save(kit.ws, localagent.Agent('local-worker', 'qwen', identity=worker,
                                           profile='coding', project=str(primary)))
    task = TaskBoard(kit.ws).create(kit.owner, {'title': 'Task', 'description': 'Current code',
                                             'acceptance': ['Checks pass']})['id']
    scope = task_worktrees.prepare(kit.ws, parent, worker, task)
    from pathlib import Path
    target = Path(scope['project'])
    git.run(['worktree', 'add', '-b', scope['branch'], str(target), scope['baseline_commit']], cwd=primary)
    (target / 'pending.txt').write_text('Unique pending work')
    with GraphStore(kit.ws.base / 'coordination.db') as graph:
        active = task_worktrees.activate(graph, worker, task)
        assert active['state'] == 'active'
        assert task_worktrees.binding(graph, worker, task)['state'] == 'active'
    assert (target / 'pending.txt').read_text() == 'Unique pending work'
