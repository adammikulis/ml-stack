"""Local coding workers consume graph task assignments and preserve blocked outcomes."""

import time

from ml_stack.graph.store import GraphStore
from ml_stack.workspace import localagent as la, localloop, task_coding, task_runtime, tokens
from ml_stack.workspace.chain import held
from ml_stack.workspace.taskboard import TaskBoard


def assigned(ws, identity, board):
    """Return the oldest queued task with an authenticated scheduler allocation."""
    tasks = {row['id']: row for row in board.list(tokens.load(ws.base, identity))['tasks']
             if row['state'] == 'queued'}
    with held(ws.base / 'coordination.lock'), GraphStore(ws.base / 'coordination.db') as graph:
        allocations = [row['attrs'] for row in graph.nodes('allocation')
                       if row['attrs']['worker'] == identity and row['attrs']['task'] in tasks]
    return min(allocations, key=lambda row: tasks[row['task']]['created_at'], default=None)


def run(ws, name):
    """Execute scheduler-assigned tasks until the worker's maintained stop flag is set."""
    status, board = la.Status(ws, name), TaskBoard(ws)
    held = None
    try:
        while not la.stop_file(ws, name).exists():
            if la.pause_file(ws, name).exists():
                status.update(state='paused', detail='Canonical tasks wait until resumed')
                time.sleep(1)
                continue
            agent = la.load(ws, name)
            if agent is None:
                raise ValueError('the registered local worker is missing')
            if agent.harness != 'claude':
                raise ValueError('canonical coding requires the bounded Claude harness')
            identity = agent.identity or agent.name
            if held is None:
                held = localloop.lease_model(agent)
                status.update(lease=held.lease)
            allocation = assigned(ws, identity, board)
            if not allocation:
                status.update(state='idle', detail='Waiting for a canonical task allocation')
                time.sleep(1)
                continue
            status.update(state='working', detail='Executing canonical task', task=allocation['task'])
            def native(task, project, stopped, checkpoint, runner=agent):
                return task_coding.perform(ws, runner, task, project, (stopped, checkpoint))
            outcome = task_runtime.execute(board, identity, allocation['task'], allocation['allocation_id'], native,
                                           stopped=lambda: la.stop_file(ws, name).exists())
            status.update(state=outcome.get('state', 'review'), detail=outcome.get('blocked_reason', 'Awaiting independent review'),
                          task=allocation['task'])
    finally:
        status.update(state='stopped', detail='Canonical worker stopped')
        try:
            if held is not None:
                held.release()
        except (OSError, RuntimeError) as error:
            status.update(detail=f'Canonical worker stopped; lease cleanup failed: {error}')
            raise
        status.update(lease={})
