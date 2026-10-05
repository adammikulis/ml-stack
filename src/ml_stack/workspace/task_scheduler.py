"""Registered parents allocate ready tasks to their enrolled local workers."""

import sys
import time

from ml_stack.workspace import (
    child_renewal,
    localagent as la,
    resource_allocations,
    task_worktrees,
    tokens,
)
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.service import Workspace
from ml_stack.workspace.taskboard import TaskBoard


def assign_next(ws, parent_token: str, worker: str, broker_lease: str):
    """Prepare and allocate the oldest eligible queued task without retrying blocked work."""
    parent, child = ws.auth(parent_token), ws.auth(tokens.load(ws.base, worker))
    if child.parent != parent.id:
        raise Denied('the scheduler must be the worker registered parent')
    child_renewal.renew(ws, parent_token, worker)
    board = TaskBoard(ws)
    runner = resource_allocations._worker(ws, worker)
    grant = resource_allocations._grant(broker_lease, resource_allocations.broker_wire.status(start=False))
    tasks = board.list(parent_token)['tasks']
    ready = [task for task in tasks if task['state'] == 'queued' and task['created_by'] == parent.id
             and all(dep['state'] == 'completed' for dep in task['dependencies'])
             and set(task['capabilities']) <= {runner.profile}
             and task['limits'].get('model', grant['model']) == grant['model']]
    if not ready:
        return None
    task = min(ready, key=lambda row: row['created_at'])
    task_worktrees.prepare(ws, parent_token, worker, task['id'])
    return resource_allocations.assign(ws, parent_token, worker, task['id'], broker_lease)


def main():
    """Continuously assign ready tasks using the registered parent's private token."""
    if len(sys.argv) != 3:
        return 2
    ws, parent, name = Workspace(), sys.argv[1], la.check_name(sys.argv[2])
    while not la.stop_file(ws, name).exists():
        agent = la.load(ws, name)
        if agent is None:
            raise ValueError('the local worker is missing')
        status = la.status_of(ws, name)
        if status.get('state') == 'idle' and status.get('lease', {}).get('id'):
            assign_next(ws, tokens.load(ws.base, parent), agent.identity or agent.name, status['lease']['id'])
        time.sleep(1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
