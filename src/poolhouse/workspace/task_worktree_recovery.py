"""Parent-authorized recovery of unchanged inactive task checkouts."""

from pathlib import Path
from uuid import uuid4

from poolhouse import worktreerules
from poolhouse.serve.process import pid_exists, started_at
from poolhouse.workspace import integration_git as repo, localagent
from poolhouse.workspace.identity import HUMAN, Denied
from poolhouse.workspace.task_graph import link, record, save
from poolhouse.workspace.task_schema import text
from poolhouse.workspace.taskboard import TaskBoard


def _inactive(ws, graph, task, scope):
    if task['state'] not in ('blocked', 'queued') or task.get('proposal_id'):
        raise Denied('only an unsubmitted pending task can dematerialize its checkout')
    if task.get('lease_id') and record(graph, task['lease_id'], 'lease')['active']:
        raise Denied('the task execution lease remains active')
    runners = [localagent.load(ws, name) for name in localagent.names(ws)]
    for runner in runners:
        if runner and (runner.identity or runner.name) == scope['worker'] and runner.pid \
                and pid_exists(runner.pid) and started_at(runner.pid) == runner.process_started:
            raise Denied('the assigned native worker remains live')
    for node in graph.nodes('allocation'):
        value = node['attrs']
        if value.get('task') != task['id']:
            continue
        if value.get('worker') != scope['worker']:
            raise Denied('the task has another worker allocation requiring recovery')
        if value.get('holder_pid') and pid_exists(value['holder_pid']) \
                and started_at(value['holder_pid']) == value.get('holder_started'):
            raise Denied('the assigned resource holder remains live')
        if value.get('scheduler_pid') and pid_exists(value['scheduler_pid']):
            raise Denied('the task scheduler remains live')


def dematerialize(ws, token: str, ident: str, reason: str) -> dict:
    """Remove an unchanged inactive task checkout while retaining its pending task history."""
    board, who = TaskBoard(ws), ws.auth(token)
    ws._may(who, 'claim')
    reason = text(reason, 'recovery reason', 2000)
    with board._store() as graph:
        task = board._task(graph, ident)
        scope = record(graph, 'task-worktree:' + ident.removeprefix('task:'), 'task-worktree')
        state = scope.get('state')
        if scope['task'] != ident or state not in ('active', 'reserved', None):
            raise Denied('the task has no recoverable checkout')
        parent = ws.registry.info(scope['worker']).get('parent')
        if who.role != HUMAN and not (who.id == scope['owner'] == parent):
            raise Denied('only the actual registered task parent or person can recover its checkout')
        _inactive(ws, graph, task, scope)
        source, target = Path(scope['source_project']), Path(scope['project'])
        checkout = worktreerules.checkouts(source)
        if not checkout or checkout[0] != source.resolve():
            raise Denied('the source repository changed')
        primary = checkout[1]
        with ws.claims.inactive_worktree(who, scope):
            if state is None and not target.is_dir():
                raise Denied('the legacy task requires its registered checkout for verified recovery')
            if target.exists():
                trees = worktreerules.checkouts(target)
                if not trees or trees != (target.resolve(), primary) or target.resolve() == primary:
                    raise Denied('the assigned task checkout registration changed')
                if repo.git(target, 'branch', '--show-current') != scope['branch'] \
                        or repo.git(target, 'rev-parse', 'HEAD') != scope['baseline_commit']:
                    raise Denied('the task checkout contains unique committed work')
            landed = repo.git(primary, 'rev-parse', 'HEAD')
            repo.remove_merged(primary, target, scope['branch'], landed)
        save(graph, 'task-worktree', {**scope, 'state': 'reserved', 'dematerialized_at': ws.clock()})
        event = {'id': 'checkpoint:' + uuid4().hex, 'task': ident, 'worker': scope['worker'],
                 'at': ws.clock(), 'summary': reason, 'transition': 'dematerialize',
                 'authorized_by': who.id, 'baseline_commit': scope['baseline_commit']}
        save(graph, 'checkpoint', event)
        link(graph, ident, event['id'], 'checkpoint')
    ws.audit('task.dematerialize', who.id, task=ident, worker=scope['worker'])
    return {'task': ident, 'state': task['state'], 'cleanup_verified': True}
