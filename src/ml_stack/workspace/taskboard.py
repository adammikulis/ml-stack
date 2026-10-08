"""Canonical structured tasks and linked outcomes in the workspace coordination graph."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from ml_stack.graph.store import GraphStore
from ml_stack.workspace import enforcement, integration_view, task_actions, task_scope, task_summary
from ml_stack.workspace.chain import held
from ml_stack.workspace.coordination import workspace_id
from ml_stack.workspace.device_accounts import account_for
from ml_stack.workspace.identity import HUMAN, Denied
from ml_stack.workspace.rates import RateLimited
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

    def _project_grant(self, identity: str, task: dict[str, Any], mode: str = '') -> bool:
        info = self.ws.registry.info(identity)
        registered = bool(self.ws.registry.role_of(identity))
        if (mode or enforcement.mode(self.ws, task['project'])) == 'strict':
            return bool(registered and task['project'] and info.get('project') == task['project'])
        return bool(registered and (not task['project'] or info.get('project') == task['project']))

    def _scope(self, who, task: dict[str, Any]) -> None:
        if not task_scope.eligible(self.ws, who, task):
            raise Denied('the task requires an authorized registered worker')

    def _qualifies(self, identity: str, role: str, task: dict[str, Any], mode: str) -> bool:
        worker = task.get('worker')
        if identity == worker:
            return False
        member = self._project_grant(identity, task, mode)
        if role == HUMAN or mode == 'open':
            return bool(role == HUMAN or member)
        parent = self.ws.registry.info(worker).get('parent', '') if worker else ''
        return bool(identity in task['reviewers'] and member or identity == task['created_by'] == parent)

    def _reviewer(self, who, task: dict[str, Any], *, economic: bool = True) -> None:
        self.ws._may(who, 'read')
        worker = task.get('worker')
        reviewer_account = account_for(self.ws, who.id)
        worker_account = account_for(self.ws, worker) if worker else None
        if economic and who.role != HUMAN and reviewer_account and worker_account \
                and reviewer_account['base_id'] == worker_account['base_id']:
            raise Denied('independent economic review requires a different enrolled device account')
        mode = enforcement.mode(self.ws, task['project']) if task['state'] == 'review' else 'open'
        if not self._qualifies(who.id, who.role, task, mode):
            raise Denied('independent review requires person or existing delegated project authority'
                         if mode == 'strict' else
                         'independent review requires a person or a registered project member other than the worker')

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
                raise Denied('designated identities must be registered in the task project')
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
                'artifacts': children['artifact'], 'reviews': children['review'],
                **integration_view.inspection(children, index, task.get('review_id'), str(self.ws.base))}

    def get(self, token: str, ident: str) -> dict[str, Any]:
        """Read a task with its native lease, checkpoints, artifacts and independent reviews."""
        self._auth(token)
        with self._store() as graph:
            task = self._details(self._task(graph, ident), *self._records(graph))
            return {**task, 'activity': task_summary.inspection(task, self.ws.clock())}

    def subscribe(self, token: str, ident: str, *, subscriber: str | None = None) -> dict[str, Any]:
        """Follow task status changes in the caller's inbox."""
        who = self._auth(token, 'send')
        subscriber = subscriber or who.id
        key = f'task-watch:{ident}:{subscriber}'
        with self._store() as graph:
            task = self._task(graph, ident)
            if subscriber != who.id:
                info = self.ws.registry.info(subscriber)
                if info.get('parent') != who.id or not self._project_grant(who.id, task):
                    raise Denied('task subscription requires the registered project worker parent')
            previous = next((row['attrs'] for row in graph.nodes('task-watch') if row['id'] == key), {})
            graph.upsert_node({'id': key, 'kind': 'task-watch', 'label': task['title'],
                               'attrs': {'task': ident, 'subscriber': subscriber, 'enabled': True}})
            if not previous.get('enabled'):
                self._queue_watcher_notice(graph, task, subscriber, 'subscribed')
        self.ws.audit('task.subscribe', who.id, task=ident)
        self.flush_notifications(token, ident)
        return {'task': ident, 'subscriber': subscriber, 'subscribed': True}

    def unsubscribe(self, token: str, ident: str) -> dict[str, Any]:
        """Stop task status messages for the caller."""
        who = self._auth(token, 'send')
        key = f'task-watch:{ident}:{who.id}'
        with self._store() as graph:
            task = self._task(graph, ident)
            previous = next((row['attrs'] for row in graph.nodes('task-watch') if row['id'] == key), {})
            graph.upsert_node({'id': key, 'kind': 'task-watch', 'label': ident,
                               'attrs': {'task': ident, 'subscriber': who.id, 'enabled': False}})
            for row in graph.nodes('task-notice'):
                notice = row['attrs']
                if notice.get('task') == ident and notice.get('recipient') == who.id \
                        and not notice.get('delivered'):
                    notice['cancelled'] = True
                    graph.upsert_node({'id': row['id'], 'kind': 'task-notice',
                                       'label': notice['event'], 'attrs': notice})
            if previous.get('enabled'):
                self._queue_watcher_notice(graph, task, who.id, 'unsubscribed')
        self.ws.audit('task.unsubscribe', who.id, task=ident)
        self.flush_notifications(token, ident)
        return {'task': ident, 'subscriber': who.id, 'subscribed': False}

    @staticmethod
    def _queue_notice(graph, notice: dict[str, str]) -> None:
        key = f'task-notice:{uuid4().hex}'
        graph.upsert_node({'id': key, 'kind': 'task-notice', 'label': notice['event'],
                           'attrs': {**notice, 'notice_id': key, 'delivered': False,
                                     'cancelled': False}})

    def _queue_watcher_notice(self, graph, task, actor: str, event: str) -> None:
        task['notice_seq'] = task.get('notice_seq', 0) + 1
        save(graph, 'task', task)
        body = f"{actor} {event} task {task['id']}: {task['title'][:140]}"
        for row in graph.nodes('task-watch'):
            attrs = row['attrs']
            if attrs.get('task') == task['id'] and attrs.get('enabled') \
                    and attrs.get('subscriber') != actor:
                self._queue_notice(graph, {'task': task['id'], 'recipient': attrs['subscriber'],
                                           'sender': actor, 'event': event, 'body': body,
                                           'sequence': task['notice_seq']})

    def watchers(self, ident: str) -> list[str]:
        """Return identities following the task."""
        with self._store() as graph:
            self._task(graph, ident)
            return sorted(row['attrs']['subscriber'] for row in graph.nodes('task-watch')
                          if row['attrs'].get('task') == ident and row['attrs'].get('enabled'))

    def queue_state_notice(self, graph: GraphStore, task: dict[str, Any], actor: str,
                           previous_state: str) -> None:
        if task['state'] == previous_state:
            return
        event = f"{previous_state}-{task['state']}"
        task['notice_seq'] = task.get('notice_seq', 0) + 1
        save(graph, 'task', task)
        body = f"Task {task['id']} is {task['state']}: {task['title'][:160]}"
        for row in graph.nodes('task-watch'):
            attrs = row['attrs']
            if attrs.get('task') == task['id'] and attrs.get('enabled') and attrs.get('subscriber') != actor:
                self._queue_notice(graph, {'task': task['id'], 'recipient': attrs['subscriber'],
                                           'sender': actor, 'event': event, 'body': body,
                                           'sequence': task['notice_seq']})

    def flush_notifications(self, token: str, ident: str | None = None) -> None:
        """Deliver persisted task notices and retain failures for a later retry."""
        who = self._auth(token, 'send')
        with self._store() as graph:
            notices = [row['attrs'] for row in graph.nodes('task-notice')
                       if not row['attrs'].get('delivered') and not row['attrs'].get('cancelled')
                       and (ident is None or row['attrs'].get('task') == ident)]
        notices.sort(key=lambda row: (row['task'], row['sequence'], row['notice_id']))
        blocked = set()
        for notice in notices:
            if notice['task'] in blocked:
                continue
            try:
                self.ws.send(token, notice['recipient'], 'status', notice['body'],
                             subject=f"Task {notice['event']}: {notice['task']}")
            except (Denied, OSError, ValueError, RateLimited):
                self.ws.audit('task.notice-pending', who.id, task=notice['task'],
                              subscriber=notice['recipient'])
                blocked.add(notice['task'])
                continue
            key = notice['notice_id']
            with self._store() as graph:
                current = next((row['attrs'] for row in graph.nodes('task-notice') if row['id'] == key), None)
                if current:
                    current['delivered'] = True
                    graph.upsert_node({'id': key, 'kind': 'task-notice', 'label': current['event'], 'attrs': current})

    def notify_watchers(self, token: str, ident: str) -> None:
        """Deliver any persisted task notices."""
        self.flush_notifications(token, ident)

    def list(self, token: str) -> dict[str, Any]:
        """Read task summaries and accepted outcome metrics."""
        self._auth(token)
        with self._store() as graph:
            index, related = self._records(graph)
            tasks = [self._details(node['attrs'], index, related) for node in graph.nodes('task')]
        now = self.ws.clock()
        tasks = [{**task, 'activity': task_summary.inspection(task, now)} for task in tasks]
        tasks.sort(key=lambda task: -task['created_at'])
        counts = Counter(task['state'] for task in tasks)
        return {'tasks': tasks, 'overview': task_summary.overview(tasks, now), 'metrics': {'states': dict(counts),
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
                previous_state = task['state']
                result = task_actions.claim(self.ws, graph, who, task, allocation)
                self.queue_state_notice(graph, task, who.id, previous_state)
        self.notify_watchers(token, ident)
        return result

    def heartbeat(self, token: str, ident: str) -> dict[str, Any]:
        """Renew an owned task lease only while its native resource allocation remains live."""
        return task_actions.heartbeat(self, token, ident)

    def checkpoint(self, token: str, ident: str, value: dict[str, Any]) -> dict[str, Any]:
        """Record a bounded worker checkpoint under its live task lease."""
        return task_actions.checkpoint(self, token, ident, value)

    def submit(self, token: str, ident: str, value: dict[str, Any]) -> dict[str, Any]:
        """Submit artifact hashes and claimed checks for independent review."""
        result = task_actions.submit(self, token, ident, value)
        self.notify_watchers(token, ident)
        return result

    def review(self, token: str, ident: str, decision: dict[str, Any]) -> dict[str, Any]:
        """Record an independent person or authenticated parent review of an immutable proposal."""
        result = task_actions.review(self, token, ident, decision)
        self.notify_watchers(token, ident)
        return result

    def block(self, token: str, ident: str, reason: str, *, kind: str = 'infrastructure') -> dict[str, Any]:
        """Record an authenticated active-task blockage without awarding an outcome."""
        result = task_actions.block(self, token, ident, reason, kind)
        self.notify_watchers(token, ident)
        return result

    def recover(self, token: str, ident: str, reason: str) -> dict[str, Any]:
        """Authorize expired-lease recovery while preserving its checkpoint history."""
        result = task_actions.ready(self, token, ident, reason, expired=True)
        self.notify_watchers(token, ident)
        return result

    def resume(self, token: str, ident: str, reason: str) -> dict[str, Any]:
        """Explicitly authorize retry after a blocked condition has been addressed."""
        result = task_actions.ready(self, token, ident, reason, expired=False)
        self.notify_watchers(token, ident)
        return result
