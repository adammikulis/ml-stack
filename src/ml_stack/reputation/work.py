"""Verified completion evidence in the maintained encrypted reputation graph."""

from __future__ import annotations

import hashlib
from typing import Any

from ml_stack.reputation import economy
from ml_stack.reputation.sealed import SealedGraph

MAX_EVIDENCE = 5000


class WorkLedger:
    """Agent completion credits and their immutable verification evidence."""

    def __init__(self, sealed: SealedGraph | None = None) -> None:
        self.sealed = sealed or SealedGraph()

    def record(self, evidence: dict[str, Any]) -> dict[str, Any]:
        """Store one authenticated verification, once per workspace, agent and task."""
        identity = f"{evidence['workspace']}:{evidence['agent']}:{evidence['task']}"
        ident = 'work:' + hashlib.sha256(identity.encode()).hexdigest()

        def change(graph):
            existing = next((node for node in graph.nodes('work_evidence')
                             if _matches(node, evidence, ('workspace', 'agent', 'task'))), None)
            if existing:
                return {**existing['attrs'], 'id': existing['id'], 'credited': False}
            if len(graph.nodes('work_evidence')) >= MAX_EVIDENCE:
                raise ValueError('verified-work evidence capacity reached')
            graph.upsert_node({'id': ident, 'kind': 'work_evidence', 'label': str(evidence['task']),
                               'attrs': evidence})
            agent_id = _agent_id(graph, evidence)
            graph.upsert_node({'id': agent_id, 'kind': 'work_agent', 'label': evidence['agent'],
                               'attrs': {'agent': evidence['agent'], 'workspace': evidence['workspace']}})
            graph.upsert_edge({'source': agent_id, 'target': ident, 'rel': 'verified_work'})
            _details(graph, ident, agent_id, evidence)
            return {**evidence, 'id': ident, 'credited': True}

        return self.sealed.edit(change)

    def record_contribution(self, evidence: dict[str, Any]) -> dict[str, Any]:
        """Persist one independently reviewed canonical attempt, with infrastructure excluded from ratings."""
        ident = 'contribution:' + hashlib.sha256(
            f"{evidence['workspace']}:{evidence['completion']}".encode()).hexdigest()
        def change(graph):
            previous = next((node for node in graph.nodes('work_contribution')
                             if _matches(node, evidence, ('workspace', 'completion'))), None)
            if previous:
                return {**previous['attrs'], 'id': previous['id'], 'recorded': False}
            if len(graph.nodes('work_contribution')) >= MAX_EVIDENCE:
                raise ValueError('reviewed contribution capacity reached')
            agent_id = _agent_id(graph, evidence)
            graph.upsert_node({'id': agent_id, 'kind': 'work_agent', 'label': evidence['agent'],
                               'attrs': {'agent': evidence['agent'], 'workspace': evidence['workspace']}})
            graph.upsert_node({'id': ident, 'kind': 'work_contribution', 'label': evidence['outcome'],
                               'attrs': evidence})
            graph.upsert_edge({'source': agent_id, 'target': ident, 'rel': 'reviewed_contribution'})
            _contribution_links(graph, ident, evidence)
            return {**evidence, 'id': ident, 'recorded': True}
        return self.sealed.edit(change)

    def migrate_credit_awards(self) -> int:
        """Grant immutable v1 base awards to previously verified historical evidence once."""
        def change(graph):
            count = 0
            for node in list(graph.nodes('work_evidence')):
                evidence = node['attrs']
                if evidence.get('award') is not None:
                    continue
                evidence = {**evidence, 'award': economy.assessment(
                    {'checks': evidence['checks'], 'artifacts': evidence['artifacts']})}
                graph.upsert_node({**node, 'attrs': evidence})
                agent_id = _agent_id(graph, evidence)
                _details(graph, node['id'], agent_id, evidence)
                count += 1
            return count
        return self.sealed.edit(change)

    def migrate_namespace(self, previous: str, current: str) -> int:
        """Move historical scope metadata once while preserving immutable record and award IDs."""
        if len(previous) != 64 or any(char not in '0123456789abcdef' for char in previous) \
                or not current.startswith('workspace:') or len(current) != 42 \
                or any(char not in '0123456789abcdef' for char in current[10:]):
            raise ValueError('migration needs an old path digest and canonical coordinator ID')
        def change(graph):
            _namespace_collisions(graph, previous, current)
            count = 0
            for kind in ('work_evidence', 'work_contribution', 'work_agent', 'work_task_reference'):
                for node in graph.nodes(kind):
                    attrs = node['attrs']
                    if attrs.get('workspace') == previous:
                        graph.upsert_node({**node, 'attrs': {**attrs, 'workspace': current,
                                                           'origin_workspace': previous}})
                        count += kind in ('work_evidence', 'work_contribution')
            if count:
                key = 'work-migration:' + hashlib.sha256(f'{previous}:{current}'.encode()).hexdigest()
                graph.upsert_node({'id': key, 'kind': 'work_namespace_migration', 'label': current,
                                   'attrs': {'from': previous, 'to': current, 'records': count}})
            return count
        return self.sealed.edit(change)

    def standings(self, workspace: str) -> list[dict[str, Any]]:
        """Completion counts and evidence for each agent in ``workspace``."""
        graph = self.sealed.graph()
        if graph is None:
            raise RuntimeError('verified-work reputation is locked or unreadable')
        agents: dict[str, list[dict[str, Any]]] = {}
        for node in graph.nodes('work_evidence'):
            evidence = node['attrs']
            if evidence.get('workspace') == workspace:
                agents.setdefault(evidence['agent'], []).append({'id': node['id'], **evidence})
        contributions = {}
        for node in graph.nodes('work_contribution'):
            item = node['attrs']
            if item['workspace'] == workspace:
                contributions.setdefault(item['agent'], []).append({'id': node['id'], **item})
                agents.setdefault(item['agent'], [])
        return [{'agent': agent, 'verified_tasks': len(evidence),
                 'evidence': sorted(evidence, key=lambda item: -item['verified_at']),
                 'contributions': contributions.get(agent, []),
                 **economy.summary(evidence, contributions.get(agent, []))}
                for agent, evidence in sorted(agents.items())]


def _details(graph, ident, agent_id, evidence):
    award = evidence['award']
    node = ident + ':award'
    graph.upsert_node({'id': node, 'kind': 'work_award', 'label': str(evidence['task']),
                       'attrs': {key: award[key] for key in ('policy', 'currency', 'base', 'quality_bonus', 'total')}})
    graph.upsert_edge({'source': agent_id, 'target': node, 'rel': 'earned'})
    graph.upsert_edge({'source': node, 'target': ident, 'rel': 'justified_by'})
    decision = ident + ':verification'
    graph.upsert_node({'id': decision, 'kind': 'work_verification', 'label': evidence['verifier'],
                       'attrs': {'reviewer': evidence['verifier'], 'at': evidence['verified_at'],
                                 'checks': evidence['checks'], 'task': evidence['task'],
                                 'completion': evidence['completion'],
                                 **{key: evidence[key] for key in ('task_hash', 'completion_hash',
                                                                  'proposal_id', 'proposal_hash', 'source')
                                    if key in evidence}}})
    graph.upsert_edge({'source': decision, 'target': ident, 'rel': 'verifies'})
    task_key = 'work-task:' + hashlib.sha256(
        f"{evidence['workspace']}:{evidence['task']}".encode()).hexdigest()
    graph.upsert_node({'id': task_key, 'kind': 'work_task_reference', 'label': str(evidence['task']),
                       'attrs': {'workspace': evidence['workspace'], 'task': evidence['task'],
                                 'spec_hash': evidence.get('task_hash', '')}})
    graph.upsert_edge({'source': decision, 'target': task_key, 'rel': 'reviews_task'})
    graph.upsert_edge({'source': ident, 'target': task_key, 'rel': 'evidence_for'})
    for quality in award['quality']:
        key = ident + ':quality:' + quality['kind']
        graph.upsert_node({'id': key, 'kind': 'work_quality_review', 'label': quality['kind'],
                           'attrs': {**quality, 'reviewer': evidence['verifier']}})
        graph.upsert_edge({'source': key, 'target': node, 'rel': 'earns_bonus'})
        graph.upsert_edge({'source': key, 'target': ident, 'rel': 'supported_by'})
    for field, kind, relation in (('review', 'work_rating', 'assesses'),
                                   ('usage', 'work_usage', 'measures')):
        if evidence.get(field) is not None and evidence.get('source') != 'canonical-taskboard':
            key = ident + ':' + field
            graph.upsert_node({'id': key, 'kind': kind, 'label': str(evidence['task']),
                               'attrs': {**evidence[field], 'reviewer': evidence['verifier']}})
            graph.upsert_edge({'source': key, 'target': ident, 'rel': relation})


def _contribution_links(graph, ident, evidence):
    task_key = 'work-task:' + hashlib.sha256(
        f"{evidence['workspace']}:{evidence['task']}".encode()).hexdigest()
    graph.upsert_node({'id': task_key, 'kind': 'work_task_reference', 'label': evidence['task'],
                       'attrs': {'task': evidence['task'], 'workspace': evidence['workspace'],
                                 'spec_hash': evidence['task_hash']}})
    graph.upsert_edge({'source': ident, 'target': task_key, 'rel': 'reviews_task'})
    if evidence.get('review') is not None and evidence['outcome'] != 'blocked_infrastructure':
        key = ident + ':rating'
        graph.upsert_node({'id': key, 'kind': 'work_rating', 'label': evidence['verifier'],
                           'attrs': {**evidence['review'], 'reviewer': evidence['verifier']}})
        graph.upsert_edge({'source': key, 'target': ident, 'rel': 'assesses'})

    if evidence.get('usage') is not None:
        key = ident + ':usage'
        graph.upsert_node({'id': key, 'kind': 'work_usage', 'label': evidence['task'],
                           'attrs': {**evidence['usage'], 'reviewer': evidence['verifier']}})
        graph.upsert_edge({'source': key, 'target': ident, 'rel': 'measures'})


def _matches(node, evidence, fields):
    return all(node['attrs'].get(key) == evidence[key] for key in fields)


def _agent_id(graph, evidence):
    existing = next((node['id'] for node in graph.nodes('work_agent')
                     if _matches(node, evidence, ('workspace', 'agent'))), None)
    return existing or 'work-agent:' + hashlib.sha256(
        f"{evidence['workspace']}:{evidence['agent']}".encode()).hexdigest()


def _namespace_collisions(graph, previous, current):
    for kind, fields in (('work_evidence', ('agent', 'task')), ('work_contribution', ('completion',))):
        rows = graph.nodes(kind)
        targets = {tuple(node['attrs'][key] for key in fields) for node in rows
                   if node['attrs'].get('workspace') == current}
        if any(tuple(node['attrs'][key] for key in fields) in targets for node in rows
               if node['attrs'].get('workspace') == previous):
            raise ValueError('namespace migration would merge duplicate independent awards or reviews')
