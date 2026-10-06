"""Task lease transitions, checkpoint evidence and independent outcome review."""

from __future__ import annotations

from contextlib import contextmanager
from uuid import uuid4

from ml_stack.graph.store import GraphStore
from ml_stack.workspace import task_worktrees
from ml_stack.workspace.chain import held
from ml_stack.workspace.family_accounts import bind_resource
from ml_stack.workspace.identity import HUMAN, Denied
from ml_stack.workspace.resource_allocations import verified_binding
from ml_stack.workspace.task_graph import link, record, save
from ml_stack.workspace.task_schema import fingerprint, review_decision, submission, text

HEARTBEAT_S = 120


def claim(ws, graph, who, task, allocation):
    if task['state'] != 'queued':
        raise Denied('the task is already claimed or has an outcome')
    if task['failures'] > task['limits'].get('max_retries', 3):
        raise Denied('task retry budget exhausted')
    if any(record(graph, dep, 'task')['state'] != 'completed' for dep in task['deps']):
        raise Denied('task dependencies have not been independently accepted')
    required = set(task['capabilities'])
    if not required <= set(allocation.get('capabilities', [])):
        raise Denied('the worker allocation lacks a required capability')
    if task['limits'].get('model') and task['limits']['model'] != allocation['model']:
        raise Denied('the allocated model does not match the task requirement')
    if allocation.get('profile') == 'coding':
        task_worktrees.activate(graph, who.id, task['id'])
    now = ws.clock()
    task['family_account'] = bind_resource(graph, task, who.id, allocation, now)
    lease = {'id': f'task-lease:{uuid4().hex}', 'task': task['id'], 'worker': who.id,
             'allocation_id': allocation['allocation_id'], 'device_id': allocation['device_id'],
             'base_id': allocation['base_id'], 'at': now, 'heartbeat_at': now,
             'deadline': now + HEARTBEAT_S, 'active': True, 'resource': allocation}
    save(graph, 'lease', lease)
    task.update(state='working', worker=who.id, device_id=allocation['device_id'],
                base_id=allocation['base_id'], lease_id=lease['id'], started_at=now)
    if task.get('blocked_at') is not None:
        task['blocked_seconds'] += max(0, now - task.pop('blocked_at'))
    task.pop('blocked_reason', None)
    task.pop('blocked_kind', None)
    task.pop('proposal_id', None)
    task.pop('review_id', None)
    save(graph, 'task', task)
    for target, kind, relation in ((f'agent:{who.id}', 'agent', 'worked-by'),
                                    (f"device:{allocation['device_id']}", 'device', 'runs-on')):
        graph.upsert_node({'id': target, 'kind': kind, 'label': target})
        link(graph, task['id'], target, relation)
    link(graph, task['id'], lease['id'], 'leased-under')
    link(graph, lease['id'], allocation['allocation_id'], 'resource-allocation')
    return lease


@contextmanager
def working(board, token, ident):
    who = board._auth(token, 'claim')
    with held(board.ws.base / 'coordination.lock'):
        with GraphStore(board.ws.base / 'coordination.db') as graph:
            task = board._task(graph, ident)
            if task['state'] != 'working' or task.get('worker') != who.id:
                raise Denied('only the current task worker may change a working task')
            lease = record(graph, task['lease_id'], 'lease')
            if not lease['active'] or lease['deadline'] <= board.ws.clock():
                raise Denied('the task lease expired; it must be recovered before work continues')
            maximum = task['limits'].get('max_wall_s')
            if maximum and board.ws.clock() - lease['at'] > maximum:
                raise Denied('task wall time budget exceeded')
        verified_binding(board.ws, who.id, ident, lease['allocation_id'])
        with GraphStore(board.ws.base / 'coordination.db') as graph, graph.transaction():
            yield graph, task, lease


def heartbeat(board, token, ident):
    with working(board, token, ident) as (graph, _task, lease):
        lease.update(heartbeat_at=board.ws.clock(), deadline=board.ws.clock() + HEARTBEAT_S)
        save(graph, 'lease', lease)
        return lease


def checkpoint(board, token, ident, value):
    if type(value) is not dict or set(value) - {'summary', 'progress', 'commit', 'environment'}:
        raise ValueError('unsupported checkpoint fields')
    fields = {key: text(item, key, 2000) for key, item in value.items()}
    if not fields.get('summary'):
        raise ValueError('a checkpoint summary is required')
    with working(board, token, ident) as (graph, task, lease):
        checkpoint = {'id': f'checkpoint:{uuid4().hex}', 'task': ident, 'worker': task['worker'],
                      'at': board.ws.clock(), 'lease_id': lease['id'], **fields}
        save(graph, 'checkpoint', checkpoint)
        link(graph, ident, checkpoint['id'], 'checkpoint')
        return checkpoint


def submit(board, token, ident, value):
    value = submission(value)
    with working(board, token, ident) as (graph, task, lease):
        proposal = {'id': f'proposal:{uuid4().hex}', 'task': ident, 'worker': task['worker'],
                    'at': board.ws.clock(), 'lease_id': lease['id'], **value}
        proposal['claimed_provenance'] = proposal['provenance']
        proposal['provenance'] = {**proposal['provenance'], 'model': lease['resource']['model'],
                                  'allocation_id': lease['allocation_id'],
                                  'device_id': lease['resource'].get('device_id'),
                                  'runtime': lease['resource'].get('harness') or lease['resource'].get('profile', '')}
        proposal['family_account'] = bind_resource(graph, task, task['worker'], lease['resource'], proposal['at'])
        proposal['proposal_hash'] = fingerprint(proposal)
        save(graph, 'proposal', proposal)
        link(graph, ident, proposal['id'], 'proposed-outcome')
        for name, digest in value['artifacts'].items():
            artifact = {'id': f'artifact:{uuid4().hex}', 'task': ident, 'proposal_id': proposal['id'],
                        'name': name, 'sha256': digest, 'at': proposal['at'], 'worker': task['worker']}
            save(graph, 'artifact', artifact)
            link(graph, proposal['id'], artifact['id'], 'produced-artifact')
        lease['active'] = False
        save(graph, 'lease', lease)
        task.update(state='review', proposal_id=proposal['id'])
        save(graph, 'task', task)
        return proposal


def review(board, token, ident, decision):
    who = board._auth(token, 'send')
    with board._store() as graph:
        task = board._task(graph, ident)
        if task['state'] != 'review':
            raise ValueError('the task has no proposal awaiting independent review')
        worker = task['worker']
        board._reviewer(who, task)
        proposal = record(graph, task['proposal_id'], 'proposal')
        if fingerprint({key: item for key, item in proposal.items() if key != 'proposal_hash'}) != proposal['proposal_hash']:
            raise ValueError('the recorded proposal hash does not match its content')
        decision = review_decision(decision, task['acceptance'], proposal['artifacts'])
        outcome = {'id': f'review:{uuid4().hex}', 'task': ident, 'worker': worker, 'verifier': who.id,
                   'at': board.ws.clock(), 'reviewed_at': board.ws.clock(),
                   'proposal_id': proposal['id'], 'proposal_hash': proposal['proposal_hash'], **decision}
        outcome['review_hash'] = fingerprint(outcome)
        save(graph, 'review', outcome)
        link(graph, ident, outcome['id'], 'independent-review')
        link(graph, outcome['id'], proposal['id'], 'reviews-proposal')
        graph.upsert_node({'id': f'agent:{who.id}', 'kind': 'agent', 'label': who.id})
        link(graph, outcome['id'], f'agent:{who.id}', 'reviewed-by')
        scope = next((node['attrs'] for node in graph.nodes('task-worktree')
                      if node['attrs']['task'] == ident), None)
        task.update(state=('accepted' if scope else 'completed') if decision['accepted'] else
                    'blocked' if decision['outcome'] == 'blocked_infrastructure' else 'rejected',
                    review_id=outcome['id'])
        if decision['outcome'] == 'rejected':
            task['failures'] += 1
        elif decision['outcome'] == 'blocked_infrastructure':
            task.update(blocked_reason=decision['reason'], blocked_at=board.ws.clock())
        save(graph, 'task', task)
        return outcome


def block(board, token, ident, reason, kind):
    who = board._auth(token, 'claim')
    reason = text(reason, 'blocked reason', 2000)
    if kind not in ('infrastructure', 'failure'):
        raise ValueError('blocked kind must be infrastructure or failure')
    with board._store() as graph:
        task = board._task(graph, ident)
        worker = task.get('worker')
        parent = board.ws.registry.info(worker).get('parent', '') if worker else ''
        if who.role != HUMAN and who.id != worker and not (who.id == task['created_by'] == parent):
            raise Denied('only the worker, person or actual task parent may record blockage')
        if task['state'] not in ('working', 'review'):
            raise ValueError('only an active task may be blocked')
        task.update(state='blocked', blocked_reason=reason, blocked_kind=kind, blocked_at=board.ws.clock())
        task['failures'] += int(kind == 'failure')
        lease = record(graph, task['lease_id'], 'lease')
        lease['active'] = False
        save(graph, 'lease', lease)
        save(graph, 'task', task)
        return task


def ready(board, token, ident, reason, *, expired):
    who = board._auth(token, 'send')
    reason = text(reason, 'resume reason', 2000)
    with board._store() as graph:
        task = board._task(graph, ident)
        board._reviewer(who, task, economic=False)
        if expired:
            if task['state'] != 'working':
                raise ValueError('only a working task can recover an expired lease')
            lease = record(graph, task['lease_id'], 'lease')
            if lease['deadline'] > board.ws.clock():
                raise Denied('a live task lease cannot be recovered')
            task['failures'] += 1
        elif task['state'] != 'blocked':
            raise ValueError('only an explicitly blocked task can resume')
        else:
            lease = record(graph, task['lease_id'], 'lease')
        if task['failures'] > task['limits'].get('max_retries', 3):
            raise Denied('task retry budget exhausted')
        lease['active'] = False
        save(graph, 'lease', lease)
        if task.get('blocked_at') is not None:
            task['blocked_seconds'] += max(0, board.ws.clock() - task.pop('blocked_at'))
        entry = {'id': f'checkpoint:{uuid4().hex}', 'task': ident, 'worker': task['worker'],
                 'at': board.ws.clock(), 'summary': reason, 'transition': 'recover' if expired else 'resume',
                 'authorized_by': who.id, 'lease_id': lease['id']}
        save(graph, 'checkpoint', entry)
        link(graph, ident, entry['id'], 'checkpoint')
        task.update(state='queued', resumed_by=who.id, resume_reason=reason)
        task.pop('blocked_reason', None)
        task.pop('blocked_kind', None)
        save(graph, 'task', task)
        return task
