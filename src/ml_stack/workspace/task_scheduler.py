"""Registered parents allocate ready tasks to their enrolled local workers."""

import os
import time

from ml_stack import activity
from ml_stack.graph.store import GraphStore
from ml_stack.serve.process import pid_exists, started_at
from ml_stack.workspace import (
    child_renewal,
    localagent as la,
    resource_allocations,
    task_authority,
    task_integration,
    task_scope,
    tokens,
)
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.task_graph import link, save
from ml_stack.workspace.taskboard import TaskBoard


def assign_next(ws, parent_token: str, worker: str, broker_lease: str, *, task=""):
    """Prepare and allocate the oldest eligible queued task without retrying blocked work."""
    parent, child = ws.auth(parent_token), ws.auth(tokens.load(ws.base, worker))
    if child.parent != parent.id:
        task_authority.authorize(ws, parent_token, worker, task)
    expiry = ws.registry.info(worker).get('expires', 0)
    if child.parent == parent.id and expiry and expiry - ws.registry.clock() < ws.limits.child_ttl_s / 4:
        child_renewal.renew(ws, parent_token, worker)
    board = TaskBoard(ws)
    runner = resource_allocations._worker(ws, worker)
    grant = resource_allocations._grant(broker_lease, resource_allocations.broker_wire.status(start=False))
    tasks = board.list(parent_token)['tasks']
    ready = [row for row in tasks if (not task or row['id'] == task) and row['state'] == 'queued' and task_scope.eligible(ws, child, row)
             and all(dep['state'] == 'completed' for dep in row['dependencies'])
             and set(row['capabilities']) <= {runner.profile}
             and row['limits'].get('model', grant['model']) == grant['model']]
    if not ready:
        return None
    task = min(ready, key=lambda row: row['created_at'])
    return resource_allocations.assign(ws, parent_token, worker, task['id'], broker_lease)



def integrate_completed(ws, parent_token, worker):
    """Attempt each independently accepted worker review once and preserve its outcome."""
    parent = ws.auth(parent_token)
    if ws.registry.info(worker).get('parent') != parent.id or not ws.registry.role_of(worker):
        raise Denied('integration polling requires the registered worker parent')
    board = TaskBoard(ws)
    results = []
    for summary in board.list(parent_token)['tasks']:
        if summary['state'] != 'accepted' or summary.get('worker') != worker:
            continue
        task = board.get(parent_token, summary['id'])
        review = task.get('review') or {}
        if not review.get('accepted') or review.get('outcome') != 'accepted':
            continue
        ident = 'integration-attempt:' + review['review_hash']
        attempt = {'id': ident, 'task': task['id'], 'worker': worker, 'owner': parent.id,
                   'review_hash': review['review_hash'], 'state': 'started', 'started': ws.clock(),
                   'owner_pid': os.getpid(), 'owner_started': started_at(os.getpid())}
        with held(ws.base / 'coordination.lock'), GraphStore(ws.base / 'coordination.db') as graph:
            old = next((node['attrs'] for node in graph.nodes('integration-attempt') if node['id'] == ident), None)
            if old:
                live = bool(old.get('owner_pid') and old.get('owner_started')
                            and pid_exists(old['owner_pid'])
                            and started_at(old['owner_pid']) == old['owner_started'])
                if old['state'] == 'started' and old['owner'] == parent.id and not live:
                    outcome = {'state': 'blocked', 'reason':
                        'The previous integration owner is no longer live; preserve and inspect its candidate before recovery.'}
                    old.update(state='blocked', outcome=outcome, finished=ws.clock(), recovery_required=True)
                    save(graph, 'integration-attempt', old)
                    results.append(outcome)
                continue
            save(graph, 'integration-attempt', attempt)
            link(graph, ident, task['id'], 'integrates-task')
            link(graph, ident, review['id'], 'accepted-review')
        try:
            outcome = task_integration.integrate(ws, parent_token, task['id'])
        except (Denied, OSError, ValueError, RuntimeError) as error:
            outcome = {'state': 'blocked', 'reason': str(error)}
        attempt.update(state=outcome['state'], outcome=outcome, finished=ws.clock())
        with held(ws.base / 'coordination.lock'), GraphStore(ws.base / 'coordination.db') as graph:
            save(graph, 'integration-attempt', attempt)
        activity.record('agent.integration', actor=parent.id, subject=task['title'],
                        outcome=attempt['state'], refs={'task': task['id'], 'worker': worker},
                        meta={'review_hash': review['review_hash'], 'reason': outcome.get('reason', '')})
        results.append(outcome)
    return results


def watch(ws, parent_token, name, *, task=""):
    """Continuously assign ready tasks using the registered parent's private token."""
    while not la.stop_file(ws, name).exists():
        agent = la.load(ws, name)
        if agent is None:
            raise ValueError('the local worker is missing')
        if not task:
            integrate_completed(ws, parent_token, agent.identity or agent.name)
        status = la.status_of(ws, name)
        if status.get('state') == 'idle' and status.get('lease', {}).get('id'):
            assign_next(ws, parent_token, agent.identity or agent.name, status['lease']['id'], task=task)
        time.sleep(1)
    return 0
