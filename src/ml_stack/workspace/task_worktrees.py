"""Parent-owned task worktrees preserve an immutable repository baseline."""

from pathlib import Path

from ml_stack.graph.store import GraphStore
from ml_stack.net import git
from ml_stack.workspace import localagent, task_scope, tokens
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import HUMAN, Denied
from ml_stack.workspace.task_schema import TASK_ID


def prepare(ws, token: str, worker: str, task: str) -> dict:
    """Create a claimed isolated worktree for one authorized canonical task."""
    caller, child = ws.auth(token), ws.auth(tokens.load(ws.base, worker))
    if caller.role != HUMAN and child.parent != caller.id:
        raise Denied('only the person or registered worker parent prepares task worktrees')
    if not TASK_ID.fullmatch(task):
        raise ValueError('a canonical task ID is required')
    runners = [localagent.load(ws, name) for name in localagent.names(ws)]
    matches = [row for row in runners if row and (row.identity or row.name) == worker]
    if len(matches) != 1:
        raise Denied('one registered local worker is required')
    source = Path(matches[0].project).resolve(strict=True)
    if not (source / '.git').is_file():
        raise Denied('the configured source must be an isolated git worktree')
    suffix = task.split(':')[1]
    target, branch = source.parent / f'task-{suffix}', f'{caller.id}/task-{suffix}'
    key = f'task-worktree:{suffix}'
    with held(ws.base / 'coordination.lock'), GraphStore(ws.base / 'coordination.db') as graph:
        spec = next((row['attrs'] for row in graph.nodes('task') if row['id'] == task), None)
        if not spec or spec['state'] != 'queued' or not task_scope.eligible(ws, child, spec):
            raise Denied('the worker must be eligible for the queued canonical task')
        old = next((row['attrs'] for row in graph.nodes('task-worktree') if row['id'] == key), None)
        if old:
            if old['worker'] != worker or old['source_project'] != str(source):
                raise Denied('the task worktree belongs to another assignment')
            return old
        if target.exists():
            raise Denied('the task worktree path already exists without an assignment')
        baseline = git.head(source)
        ws.claim(token, 'worktree', str(target), ttl_s=86400, note=f'Canonical task {task}')
        ws.claim(token, 'branch', branch, ttl_s=86400, note=f'Canonical task {task}')
        git.run(['worktree', 'add', '-b', branch, str(target), baseline], cwd=source)
        record = {'id': key, 'task': task, 'worker': worker, 'owner': caller.id,
                  'project': str(target), 'source_project': str(source), 'baseline_commit': baseline,
                  'branch': branch}
        graph.upsert_node({'id': key, 'kind': 'task-worktree', 'label': task, 'attrs': record})
        graph.upsert_edge({'source': key, 'target': task, 'rel': 'execution-scope'})
        return record


def binding(graph, worker: str, task: str) -> dict:
    """Return the task's existing trusted worktree assignment."""
    found = [row['attrs'] for row in graph.nodes('task-worktree')
             if row['attrs']['task'] == task and row['attrs']['worker'] == worker]
    if len(found) != 1 or not (Path(found[0]['project']) / '.git').is_file():
        raise Denied('the task has no prepared isolated worktree')
    return found[0]
