"""Local coding workers consume graph task assignments and preserve blocked outcomes."""

import time

from poolhouse.graph.store import GraphStore
from poolhouse.workspace import localagent as la, localloop, task_coding, task_runtime, tokens
from poolhouse.workspace.chain import held
from poolhouse.workspace.taskboard import TaskBoard


def assigned(ws, identity, board, broker_lease):
    """Return the oldest queued task with an authenticated scheduler allocation."""
    tasks = {row['id']: row for row in board.list(tokens.load(ws.base, identity))['tasks']
             if row['state'] == 'queued'}
    with held(ws.base / 'coordination.lock'), GraphStore(ws.base / 'coordination.db') as graph:
        allocations = [row['attrs'] for row in graph.nodes('allocation')
                       if row['attrs']['worker'] == identity and row['attrs']['task'] in tasks
                       and row['attrs']['lease_id'] == broker_lease]
    return min(allocations, key=lambda row: tasks[row['task']]['created_at'], default=None)


def run(ws, name):
    """Execute scheduler-assigned tasks until the worker's maintained stop flag is set."""
    status, board = la.Status(ws, name), TaskBoard(ws)
    held = None
    try:
        while not la.stop_file(ws, name).exists():
            if la.pause_file(ws, name).exists():
                status.update(state='paused', detail='Project tasks wait until resumed')
                time.sleep(1)
                continue
            agent = la.load(ws, name)
            if agent is None:
                raise ValueError('the registered local worker is missing')
            if agent.harness not in ('pi', 'claude'):
                raise ValueError('project task work needs a supported coding harness')
            identity = agent.identity or agent.name
            if held is None:
                held = localloop.lease_model(agent)
                status.update(lease=held.lease)
            allocation = assigned(ws, identity, board, held.lease['id'])
            if not allocation:
                status.update(state='idle', detail='Waiting for a task assignment')
                time.sleep(1)
                continue
            status.update(state='working', detail='Executing project task', task=allocation['task'])
            def native(task, project, stopped, checkpoint, runner=agent):
                return task_coding.perform(ws, runner, task, project, (stopped, checkpoint))
            outcome = task_runtime.execute(board, identity, allocation['task'], allocation['allocation_id'], native,
                                           stopped=lambda: la.stop_file(ws, name).exists(),
                                           wall_s=localloop.caps_of(agent).seconds)
            status.update(state=outcome.get('state', 'review'), detail=outcome.get('blocked_reason', 'Awaiting independent review'),
                          task=allocation['task'])
    finally:
        status.update(state='stopped', detail='Project worker stopped')
        try:
            if held is not None:
                held.release()
        except (OSError, RuntimeError) as error:
            status.update(detail=f'Project worker stopped; lease cleanup failed: {error}')
            raise
        status.update(lease={})
