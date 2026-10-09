"""Live task authority for managing existing project workers."""

from poolhouse import worktreerules
from poolhouse.graph.store import GraphStore
from poolhouse.workspace import localagent as la, project, tokens
from poolhouse.workspace.chain import held
from poolhouse.workspace.identity import HUMAN, Denied
from poolhouse.workspace.task_graph import record
from poolhouse.workspace.task_schema import SPEC_FIELDS, TASK_ID, fingerprint


def authorize(ws, token, worker, task):
    """Return the saved worker after checking the caller's live task authority."""
    caller = ws.auth(token)
    ws._may(caller, "send")
    runners = [la.load(ws, name) for name in la.names(ws)]
    matches = [row for row in runners if row and worker in (row.name, row.identity or row.name)]
    if len(matches) != 1:
        raise Denied('task management requires one existing local worker')
    runner = matches[0]
    child = ws.auth(tokens.load(ws.base, runner.identity or runner.name))
    ws._may(caller, 'read')
    if caller.role == HUMAN or child.role == HUMAN or caller.id == child.id:
        raise Denied('task management requires separate authenticated agents and a registered parent or task creator')
    if type(task) is not str or not TASK_ID.fullmatch(task):
        raise ValueError('a task ID is required')
    with held(ws.base / 'coordination.lock'), GraphStore(ws.base / 'coordination.db') as graph:
        spec = record(graph, task, 'task')
        if fingerprint({key: spec[key] for key in SPEC_FIELDS}) != spec['spec_hash']:
            raise ValueError('the task specification changed after creation')
    caller_info, child_info = ws.registry.info(caller.id), ws.registry.info(child.id)
    if spec['created_by'] != caller.id or child.id not in spec['assignees']:
        raise Denied('only the registered worker parent or task creator manages an explicitly assigned worker')
    if not spec['project'] or caller_info.get('project') != spec['project'] \
            or child_info.get('project') != spec['project']:
        raise Denied('task management requires live matching person-set project grants')
    source = la.check_project(runner.project, ws.base)
    if not source or project.describe(source).get('key') != spec['project'].get('key') or not worktreerules.checkouts(source):
        raise Denied('the saved worker source must match the granted task repository')
    if caller.id not in runner.orders_from:
        raise Denied('the worker must already take orders from the task creator')
    status = la.status_of(ws, runner.name)
    if status.get('state') == 'working' and status.get('task') and status['task'] != task:
        raise Denied('the worker is executing another task')
    if spec['state'] not in ('queued', 'working', 'blocked', 'review', 'accepted'):
        raise Denied('the task is no longer active')
    return runner, spec
