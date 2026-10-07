"""Authenticated relocation of inactive canonical task source bindings."""

from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from ml_stack import worktreerules
from ml_stack.command import flag
from ml_stack.graph.store import GraphStore
from ml_stack.workspace import integration_git as repo, localagent, tokens
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import HUMAN, Denied, Identity
from ml_stack.workspace.task_graph import link, record, save
from ml_stack.workspace.task_schema import text
from ml_stack.workspace.task_worktree_recovery import _inactive, dematerialize
from ml_stack.workspace.taskboard import TaskBoard


@dataclass(frozen=True)
class Binding:
    who: Identity
    child: Identity
    source: str
    target: Path


def _worker(ws, token, worker):
    who = ws.auth(token)
    ws._may(who, 'claim')
    matches = [row for name in localagent.names(ws)
               if (row := localagent.load(ws, name)) and worker in (row.name, row.identity or row.name)]
    if len(matches) != 1:
        raise Denied('source recovery requires one existing local worker')
    runner = matches[0]
    child = ws.auth(tokens.load(ws.base, runner.identity or runner.name))
    if who.role == HUMAN or child.role == HUMAN or child.parent != who.id:
        raise Denied('only the authenticated registered worker parent can relocate its source')
    return who, runner, child


def _scopes(ws, graph, board, binding):
    who, child, source, target = binding.who, binding.child, binding.source, binding.target
    scopes = sorted([node['attrs'] for node in graph.nodes('task-worktree')
                     if node['attrs']['worker'] == child.id and node['attrs']['source_project'] == source],
                    key=lambda scope: scope['task'])
    if not scopes:
        raise Denied('the worker has no pending canonical source bindings')
    for scope in scopes:
        task = board._task(graph, scope['task'])
        if who.id != scope['owner'] or who.id != task['created_by']:
            raise Denied('source recovery requires ownership and creation of every affected task')
        if task['project'] and any(ws.registry.info(ident).get('project') != task['project']
                                   for ident in (who.id, child.id)):
            raise Denied('source recovery cannot change existing task project grants')
        _inactive(ws, graph, task, scope)
        if scope.get('state') != 'reserved' or Path(scope['project']).exists() or Path(scope['project']).is_symlink():
            raise Denied('dematerialize every affected task checkout before relocating its source')
        if not repo.ancestor(target, scope['baseline_commit'], repo.git(target, 'rev-parse', 'HEAD')):
            raise Denied('the replacement source does not contain the original task baseline')
    return scopes


def _source(source, target):
    old, new = worktreerules.checkouts(source), worktreerules.checkouts(target)
    if not old or old[0] != Path(source).resolve() or not new or new[0] != target or old[1] != new[1]:
        raise Denied('the replacement must be a checkout of the same physical Git repository')
    repo.clean(target)


def _pending(graph, worker, target):
    rows = [node['attrs'] for node in graph.nodes('source-recovery')
            if node['attrs']['worker'] == worker and node['attrs']['target'] == target]
    return max(rows, key=lambda row: row['at']) if rows else None


def _apply(ws, journal, runner, scopes):
    changed = replace(runner, project=journal['target'])
    try:
        localagent.save(ws, changed)
        with GraphStore(ws.base / 'coordination.db') as graph, graph.transaction():
            for scope in scopes:
                current = record(graph, scope['id'], 'task-worktree')
                if current != scope:
                    raise Denied('the task binding changed during source recovery')
                save(graph, 'task-worktree', {**scope, 'source_project': journal['target']})
                event = {'id': 'checkpoint:' + uuid4().hex, 'task': scope['task'],
                         'worker': scope['worker'], 'at': ws.clock(), 'summary': journal['reason'],
                         'transition': 'rebind-source', 'authorized_by': journal['owner'],
                         'source_project': journal['source'], 'target_project': journal['target'],
                         'baseline_commit': scope['baseline_commit']}
                save(graph, 'checkpoint', event)
                link(graph, scope['task'], event['id'], 'checkpoint')
            save(graph, 'source-recovery', {**journal, 'state': 'committed'})
    except Exception:
        localagent.save(ws, runner)
        raise


def rebind(ws, token: str, worker: str, project: str, reason: str) -> dict:
    """Relocate one stopped worker's pending source scopes within its physical Git repository."""
    reason = text(reason, 'recovery reason', 2000)
    if not project:
        raise ValueError('a replacement source checkout is required')
    target = Path(localagent.check_project(project, ws.base))
    board = TaskBoard(ws)
    with held(localagent.folder(ws) / 'start.lock'), held(ws.registry.path.with_name('agents.lock')), \
            held(ws.base / 'coordination.lock'):
        who, runner, child = _worker(ws, token, worker)
        with GraphStore(ws.base / 'coordination.db') as graph, graph.transaction():
            previous = _pending(graph, child.id, str(target))
            if previous and previous['owner'] != who.id:
                raise Denied('the existing source recovery belongs to another authenticated parent')
            if previous and previous['state'] == 'committed' and runner.project == str(target):
                checkout = worktreerules.checkouts(target)
                if not checkout or checkout[0] != target or str(checkout[1]) != previous['primary']:
                    raise Denied('the completed source recovery repository changed')
                repo.clean(target)
                scopes = _scopes(ws, graph, board, Binding(who, child, str(target), target))
                if previous['tasks'] != [scope['task'] for scope in scopes]:
                    raise Denied('the completed source recovery task bindings changed')
                if any(node['attrs']['worker'] == child.id and
                       node['attrs']['source_project'] == previous['source']
                       for node in graph.nodes('task-worktree')):
                    raise Denied('the worker still has task scopes bound to its old source')
                return {'worker': child.id, 'project': str(target), 'tasks': previous['tasks'],
                        'recovery': previous['id'], 'verified': True}
            source = previous['source'] if previous and previous['state'] == 'prepared' else runner.project
            if source == str(target) or runner.project not in (source, str(target)):
                raise Denied('the saved worker source changed outside its recorded recovery')
            _source(source, target)
            scopes = _scopes(ws, graph, board, Binding(who, child, source, target))
            snapshot = runner.as_dict()
            snapshot['project'] = source
            if previous and previous['state'] == 'prepared':
                if previous['worker_config'] != snapshot or previous['tasks'] != [s['task'] for s in scopes]:
                    raise Denied('the prepared source recovery configuration changed')
                journal = previous
            else:
                journal = {'id': 'source-recovery:' + uuid4().hex, 'worker': child.id,
                           'owner': who.id, 'source': source, 'target': str(target),
                           'primary': str(worktreerules.checkouts(target)[1]),
                           'reason': reason, 'at': ws.clock(), 'state': 'prepared',
                           'tasks': [scope['task'] for scope in scopes], 'worker_config': snapshot}
                save(graph, 'source-recovery', journal)
        _apply(ws, journal, runner, scopes)
    ws.audit('task.rebind-source', who.id, worker=child.id, recovery=journal['id'])
    return {'worker': child.id, 'project': str(target), 'tasks': journal['tasks'],
            'recovery': journal['id'], 'verified': True}


TABLE = [
    ("task-rebind-source", "relocate a stopped worker source, preserving pending task grants", [flag("worker"), flag("project"), flag("reason")],
     lambda a, w, t: rebind(w, t, a.worker, a.project, a.reason)),
    ("task-dematerialize", "remove an unchanged inactive task checkout, preserving its pending history", [flag("id"), flag("reason")],
     lambda a, w, t: dematerialize(w, t, a.id, a.reason)),
]
