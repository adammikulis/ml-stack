"""Canonical structured tasks and linked outcomes in the workspace coordination graph."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from ml_stack.graph.store import GraphStore
from ml_stack.workspace import task_actions
from ml_stack.workspace.chain import held
from ml_stack.workspace.coordination import workspace_id
from ml_stack.workspace.identity import HUMAN, Denied
from ml_stack.workspace.resource_allocations import verified_binding
from ml_stack.workspace.task_graph import link, record, save
from ml_stack.workspace.task_schema import SPEC_FIELDS, TASK_ID, fingerprint, task_spec

MAX_TASKS = 5000



class TaskBoard:
    """Authenticated task state, atomic worker claims and independently reviewed outcomes."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.workspace_id = workspace_id(ws)

    def _auth(self, token: str, capability: str = 'read'):
        who = self.ws.auth(token)
        self.ws._may(who, capability)
        return who

    @contextmanager
    def _store(self):
        with held(self.ws.base / 'coordination.lock'), GraphStore(self.ws.base / 'coordination.db') as graph, graph.transaction():
            yield graph

    def _task(self, graph: GraphStore, ident: str) -> dict[str, Any]:
        if type(ident) is not str or not TASK_ID.fullmatch(ident):
            raise ValueError('a canonical task ID is required')
        task = record(graph, ident, 'task')
        if fingerprint({key: task[key] for key in SPEC_FIELDS}) != task['spec_hash']:
            raise ValueError('the task specification changed after creation')
        return task

    def _project_grant(self, identity: str, task: dict[str, Any]) -> bool:
        info = self.ws.registry.info(identity)
        return bool(self.ws.registry.role_of(identity) and task['project']
                    and info.get('project') == task['project'])

    def _scope(self, who, task: dict[str, Any]) -> None:
        creator = self.ws.registry.info(task['created_by'])
        delegated = who.id in task['assignees'] and self._project_grant(who.id, task)
        if who.role == HUMAN or not (delegated or who.parent == task['created_by']
                                    or (creator['role'] == HUMAN and not task['assignees'])):
            raise Denied('the task requires an authorized registered worker')

    def _reviewer(self, who, task: dict[str, Any]) -> None:
        worker = task.get('worker')
        parent = self.ws.registry.info(worker).get('parent', '') if worker else ''
        designated = who.id in task['reviewers'] and self._project_grant(who.id, task)
        if who.id == worker or not (who.role == HUMAN or designated
                                   or who.id == task['created_by'] == parent):
            raise Denied('independent review requires person or existing delegated project authority')

    def assert_reviewer(self, token: str, ident: str) -> dict[str, Any]:
        """Recheck live reviewer authority independently of stored review claims."""
        who = self._auth(token, 'send')
        with self._store() as graph:
            task = self._task(graph, ident)
            self._reviewer(who, task)
            return task

    def create(self, token: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Create a bounded structured task or reuse its identical source-key record."""
        who, spec = self._auth(token, 'send'), task_spec(spec)
        for identity in spec['assignees'] + spec['reviewers']:
            if not self.ws.registry.role_of(identity) or not self._project_grant(identity, spec):
                raise Denied('designated identities require existing person-set project grants')
        if (spec['assignees'] or spec['reviewers']) and who.role != HUMAN and not self._project_grant(who.id, spec):
            raise Denied('the task creator lacks delegated project authority')
        with self._store() as graph:
            rows = graph.nodes('task')
            previous = next((node['attrs'] for node in rows if spec['source_key']
                             and node['attrs']['source_key'] == spec['source_key']), None)
            if previous:
                if previous['created_by'] != who.id or previous['spec_hash'] != fingerprint(spec):
                    raise ValueError('source key already belongs to another task specification')
                return dict(previous)
            if len(rows) >= MAX_TASKS:
                raise ValueError('task board capacity reached')
            for dependency in spec['deps']:
                self._task(graph, dependency)
            task = {'id': f'task:{uuid4().hex}', **spec, 'created_by': who.id,
                    'spec_hash': fingerprint(spec), 'created_at': self.ws.clock(),
                    'state': 'queued', 'failures': 0, 'blocked_seconds': 0.0,
                    'workspace': self.workspace_id}
            save(graph, 'task', task)
            graph.upsert_node({'id': f'agent:{who.id}', 'kind': 'agent', 'label': who.id})
            link(graph, task['id'], f'agent:{who.id}', 'created-by')
            link(graph, task['id'], self.workspace_id, 'in-workspace')
            for dependency in spec['deps']:
                link(graph, task['id'], dependency, 'depends-on')
        self.ws.audit('task.create', who.id, task=task['id'])
        return task

    def _records(self, graph: GraphStore):
        index, related = {}, defaultdict(lambda: defaultdict(list))
        for node in graph.nodes():
            index[node['id']] = node['attrs']
            if node['attrs'].get('task'):
                related[node['attrs']['task']][node['kind']].append(node['attrs'])
        return index, related

    def _details(self, task: dict[str, Any], index, related) -> dict[str, Any]:
        children = related[task['id']]
        for rows in children.values():
            rows.sort(key=lambda row: row.get('at', 0))
        dependencies = [{'id': dep, 'title': index[dep]['title'], 'state': index[dep]['state']}
                        for dep in task['deps']]
        return {**task, 'dependencies': dependencies,
                'lease': index.get(task.get('lease_id')), 'proposal': index.get(task.get('proposal_id')),
                'review': index.get(task.get('review_id')), 'checkpoints': children['checkpoint'],
                'artifacts': children['artifact'], 'reviews': children['review']}

    def get(self, token: str, ident: str) -> dict[str, Any]:
        """Read a task with its native lease, checkpoints, artifacts and independent reviews."""
        self._auth(token)
        with self._store() as graph:
            return self._details(self._task(graph, ident), *self._records(graph))

    def list(self, token: str) -> dict[str, Any]:
        """Read task summaries and accepted outcome metrics."""
        self._auth(token)
        with self._store() as graph:
            index, related = self._records(graph)
            tasks = [self._details(node['attrs'], index, related) for node in graph.nodes('task')]
        tasks.sort(key=lambda task: -task['created_at'])
        counts = Counter(task['state'] for task in tasks)
        return {'tasks': tasks, 'metrics': {'states': dict(counts),
                'verified_outcomes': counts['completed'],
                'accepted_artifacts': sum(len(task['proposal']['artifacts']) for task in tasks
                                          if task['state'] == 'completed'),
                'failures': sum(task['failures'] for task in tasks),
                'blocked_seconds': sum(task['blocked_seconds'] + max(0, self.ws.clock() - task['blocked_at'])
                                       if task.get('blocked_at') is not None else task['blocked_seconds']
                                       for task in tasks)}}

    def claim(self, token: str, ident: str, allocation_id: str) -> dict[str, Any]:
        """Atomically claim a task against a verified scheduler resource allocation."""
        who = self._auth(token, 'claim')
        with held(self.ws.base / 'coordination.lock'):
            allocation = verified_binding(self.ws, who.id, ident, allocation_id)
            with GraphStore(self.ws.base / 'coordination.db') as graph, graph.transaction():
                task = self._task(graph, ident)
                self._scope(who, task)
                return task_actions.claim(self.ws, graph, who, task, allocation)

    def heartbeat(self, token: str, ident: str) -> dict[str, Any]:
        """Renew an owned task lease only while its native resource allocation remains live."""
        return task_actions.heartbeat(self, token, ident)

    def checkpoint(self, token: str, ident: str, value: dict[str, Any]) -> dict[str, Any]:
        """Record a bounded worker checkpoint under its live task lease."""
        return task_actions.checkpoint(self, token, ident, value)

    def submit(self, token: str, ident: str, value: dict[str, Any]) -> dict[str, Any]:
        """Submit artifact hashes and claimed checks for independent review."""
        return task_actions.submit(self, token, ident, value)

    def review(self, token: str, ident: str, decision: dict[str, Any]) -> dict[str, Any]:
        """Record an independent person or authenticated parent review of an immutable proposal."""
        return task_actions.review(self, token, ident, decision)

    def block(self, token: str, ident: str, reason: str, *, kind: str = 'infrastructure') -> dict[str, Any]:
        """Record an authenticated active-task blockage without awarding an outcome."""
        return task_actions.block(self, token, ident, reason, kind)

    def recover(self, token: str, ident: str, reason: str) -> dict[str, Any]:
        """Authorize expired-lease recovery while preserving its checkpoint history."""
        return task_actions.ready(self, token, ident, reason, expired=True)

    def resume(self, token: str, ident: str, reason: str) -> dict[str, Any]:
        """Explicitly authorize retry after a blocked condition has been addressed."""
        return task_actions.ready(self, token, ident, reason, expired=False)
