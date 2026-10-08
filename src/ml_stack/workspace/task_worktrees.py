"""Parent-owned task worktrees preserve an immutable repository baseline."""

from pathlib import Path

from ml_stack import worktreerules
from ml_stack.graph.store import GraphStore
from ml_stack.net import git
from ml_stack.workspace import (
    integration_git as repo,
    localagent,
    task_authority,
    task_scope,
    tokens,
)
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import HUMAN, Denied
from ml_stack.workspace.task_schema import TASK_ID


def prepare(ws, token: str, worker: str, task: str) -> dict:
    """Reserve a claimed checkout and committed baseline for one task."""
    caller, child = ws.auth(token), ws.auth(tokens.load(ws.base, worker))
    if caller.role != HUMAN and child.parent != caller.id:
        task_authority.authorize(ws, token, worker, task)
    if not TASK_ID.fullmatch(task):
        raise ValueError('a canonical task ID is required')
    runners = [localagent.load(ws, name) for name in localagent.names(ws)]
    matches = [row for row in runners if row and (row.identity or row.name) == worker]
    if len(matches) != 1:
        raise Denied('one registered local worker is required')
    source = Path(matches[0].project).resolve(strict=True)
    checkout = worktreerules.checkouts(source)
    if not checkout or checkout[0] != source:
        raise Denied('the configured source must be a Git checkout')
    primary = checkout[1]
    repo.clean(primary)
    baseline = repo.git(primary, 'rev-parse', 'HEAD')
    development = repo.git(primary, 'branch', '--show-current')
    repo.remote_baseline(primary, development, baseline)
    suffix = task.split(':')[1]
    target, branch = primary.parent / f'task-{suffix}', f'{caller.id}/task-{suffix}'
    key = f'task-worktree:{suffix}'
    with held(ws.base / 'coordination.lock'), GraphStore(ws.base / 'coordination.db') as graph:
        spec = next((row['attrs'] for row in graph.nodes('task') if row['id'] == task), None)
        if not spec or spec['state'] != 'queued' or not task_scope.eligible(ws, child, spec):
            raise Denied('the worker must be eligible for the queued canonical task')
        old = next((row['attrs'] for row in graph.nodes('task-worktree') if row['id'] == key), None)
        if old:
            if old['worker'] != worker or old['source_project'] != str(source):
                raise Denied('the task worktree belongs to another assignment')
            old_target = Path(old['project'])
            if old.get('state') == 'reserved' and not old_target.exists() and old['owner'] != caller.id:
                raise Denied('the reserved task must be reassigned by its recorded parent owner')
            if old.get('state') == 'reserved' and not old_target.exists() and old['baseline_commit'] != baseline:
                old = {**old, 'baseline_commit': baseline}
                graph.upsert_node({'id': key, 'kind': 'task-worktree', 'label': task, 'attrs': old})
            if old.get('state') == 'reserved' and not old_target.exists():
                ws.claim(token, 'worktree', old['project'], ttl_s=86400, note=f'Canonical task {task}')
                ws.claim(token, 'branch', old['branch'], ttl_s=86400, note=f'Canonical task {task}')
            return old
        if why := worktreerules.worktree_refusal(target, source):
            raise Denied(why)
        if target.exists():
            raise Denied('the task worktree path already exists without an assignment')
        ws.claim(token, 'worktree', str(target), ttl_s=86400, note=f'Canonical task {task}')
        ws.claim(token, 'branch', branch, ttl_s=86400, note=f'Canonical task {task}')
        record = {'id': key, 'task': task, 'worker': worker, 'owner': caller.id,
                  'project': str(target), 'source_project': str(source), 'baseline_commit': baseline,
                  'branch': branch, 'state': 'reserved'}
        graph.upsert_node({'id': key, 'kind': 'task-worktree', 'label': task, 'attrs': record})
        graph.upsert_edge({'source': key, 'target': task, 'rel': 'execution-scope'})
        return record


def binding(graph, worker: str, task: str) -> dict:
    """Return the task's existing trusted worktree assignment."""
    found = [row['attrs'] for row in graph.nodes('task-worktree')
             if row['attrs']['task'] == task and row['attrs']['worker'] == worker]
    if len(found) != 1 or found[0].get('state') not in ('reserved', 'active'):
        raise Denied('the task has no prepared isolated worktree')
    scope = found[0]
    source, target = Path(scope['source_project']), Path(scope['project'])
    checkout = worktreerules.checkouts(source)
    if not checkout or checkout[0] != source.resolve():
        raise Denied('the configured task baseline is no longer a Git checkout')
    if scope['state'] == 'active':
        checkout = worktreerules.checkouts(target)
        if not (target / '.git').is_file() or not checkout or checkout[0] != target.resolve() \
                or checkout[0] == checkout[1]:
            raise Denied('the active task checkout is missing or no longer isolated')
    return scope


def activate(graph, worker: str, task: str) -> dict:
    """Materialize the task's reserved checkout when its assigned worker claims it."""
    scope = binding(graph, worker, task)
    target, source = Path(scope['project']), Path(scope['source_project'])
    if (target / '.git').is_file():
        checkout, baseline = worktreerules.checkouts(target), worktreerules.checkouts(source)
        if not checkout or not baseline or checkout[0] != target.resolve() \
                or checkout[0] == checkout[1] or checkout[1] != baseline[1]:
            raise Denied('the existing task checkout is not registered to its assigned repository')
        if why := worktreerules.worktree_refusal(target, source, registered=True):
            raise Denied(why)
        if git.run(['branch', '--show-current'], cwd=target).stdout.strip() != scope['branch']:
            raise Denied('the task checkout branch changed before claiming')
        if not repo.ancestor(target, scope['baseline_commit'], repo.git(target, 'rev-parse', 'HEAD')):
            raise Denied('the existing task checkout diverged from its assigned baseline')
        scope = {**scope, 'state': 'active'}
        graph.upsert_node({'id': scope['id'], 'kind': 'task-worktree', 'label': task, 'attrs': scope})
        return scope
    if scope.get('state') != 'reserved':
        raise Denied('the active task checkout disappeared; recover its worktree registration')
    if target.exists():
        raise Denied('the task worktree path contains unassigned files')
    if why := worktreerules.worktree_refusal(target, source):
        raise Denied(why)
    primary = worktreerules.checkouts(source)[1]
    development = repo.git(primary, 'branch', '--show-current')
    repo.remote_baseline(primary, development, scope['baseline_commit'])
    if repo.git(primary, 'rev-parse', 'HEAD') != scope['baseline_commit']:
        raise Denied('development changed after task reservation; refresh the parent resource assignment')
    git.run(['worktree', 'add', '-b', scope['branch'], str(target), scope['baseline_commit']], cwd=source)
    scope = {**scope, 'state': 'active'}
    graph.upsert_node({'id': scope['id'], 'kind': 'task-worktree', 'label': task, 'attrs': scope})
    return scope
