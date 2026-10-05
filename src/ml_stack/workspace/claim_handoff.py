"""Explicit task-worktree delegation follows the authenticated live execution allocation."""

from pathlib import Path

from ml_stack.graph.store import GraphStore
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.resource_allocations import verified_binding
from ml_stack.workspace.task_graph import record


def _assignment(ws, who, roots):
    """Return the current task assignment for an exact launcher worktree, when one exists."""
    paths = {str(Path(root).resolve()) for root in roots}
    with GraphStore(ws.base / 'coordination.db') as graph:
        scopes = [row['attrs'] for row in graph.nodes('task-worktree') if row['attrs']['project'] in paths]
        if not scopes:
            return None
        if len(scopes) != 1:
            raise Denied('the launcher project has ambiguous task assignments')
        scope = scopes[0]
        if scope['worker'] != who.id or scope.get('owner') != who.parent:
            raise Denied('the worktree was not explicitly assigned by this worker parent')
        task = record(graph, scope['task'], 'task')
        if task['state'] != 'working' or task.get('worker') != who.id:
            raise Denied('the assigned task is not owned by this active worker')
        lease = record(graph, task['lease_id'], 'lease')
        if not lease['active'] or lease['deadline'] <= ws.clock():
            raise Denied('the assigned task lease is inactive or expired')
    allocation = verified_binding(ws, who.id, scope['task'], lease['allocation_id'])
    if allocation['project'] != scope['project'] or allocation.get('baseline_commit') != scope['baseline_commit']:
        raise Denied('the allocation does not match the trusted task worktree')
    return scope

def assignment(ws, who, roots):
    """Read the live task assignment under the coordinator ownership lock."""
    with held(ws.base / 'coordination.lock'):
        return _assignment(ws, who, roots)


def acquire(ws, who, roots):
    """Transfer only the explicitly assigned physical worktree claim to its active worker."""
    with held(ws.base / 'coordination.lock'):
        scope = _assignment(ws, who, roots)
        if scope:
            ws.claims.handoff(who, 'worktree', scope['project'], scope['owner'], scope['id'])
        return scope
