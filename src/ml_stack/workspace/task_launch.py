"""Task-scoped management of existing person-enrolled project workers."""

from dataclasses import asdict

from ml_stack.workspace import localagent as la, localloop, localstart
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.task_authority import authorize


def start(ws, token, worker, task):
    """Start the assigned worker with its saved model, identity and execution limits."""
    runner, spec = authorize(ws, token, worker, task)
    if set(spec['capabilities']) - {'coding'}:
        raise Denied('native task launch cannot satisfy the requested capabilities')
    if la.alive(runner):
        if runner.profile != 'coding' or runner.harness != 'claude':
            raise Denied('stop the existing worker before changing its task harness')
        return localstart.Started(runner.name, runner.pid, runner.model_name, runner.role, already=True)
    caps = asdict(localloop.caps_of(runner))
    ask = localstart.Ask(model=runner.model, name=runner.name, role=runner.role,
                         effort=runner.effort, max_effort=runner.max_effort, profile='coding',
                         ctx=runner.ctx, project=runner.project, orders_from=runner.orders_from, harness='claude', authority=(token, task))
    result = localstart._coding(ws, ask, None, runner.ctx, runner.project)
    ws.audit('local-agent.task-start', ws.auth(token).id, agent=worker, task=task, caps=caps)
    return result


def stop(ws, token, worker, task):
    """Stop only the existing worker assigned to the caller's active task."""
    authorize(ws, token, worker, task)
    result = localstart.stop(ws, worker)
    ws.audit('local-agent.task-stop', ws.auth(token).id, agent=worker, task=task)
    return result
