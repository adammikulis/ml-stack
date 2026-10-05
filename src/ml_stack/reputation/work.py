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
        agent_id = 'work-agent:' + hashlib.sha256(
            f"{evidence['workspace']}:{evidence['agent']}".encode()).hexdigest()

        def change(graph):
            existing = next((node for node in graph.nodes('work_evidence') if node['id'] == ident), None)
            if existing:
                return {**existing['attrs'], 'id': ident, 'credited': False}
            if len(graph.nodes('work_evidence')) >= MAX_EVIDENCE:
                raise ValueError('verified-work evidence capacity reached')
            graph.upsert_node({'id': ident, 'kind': 'work_evidence', 'label': str(evidence['task']),
                               'attrs': evidence})
            graph.upsert_node({'id': agent_id, 'kind': 'work_agent', 'label': evidence['agent'],
                               'attrs': {'agent': evidence['agent'], 'workspace': evidence['workspace']}})
            graph.upsert_edge({'source': agent_id, 'target': ident, 'rel': 'verified_work'})
            _details(graph, ident, agent_id, evidence)
            return {**evidence, 'id': ident, 'credited': True}

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
                agent_id = 'work-agent:' + hashlib.sha256(
                    f"{evidence['workspace']}:{evidence['agent']}".encode()).hexdigest()
                _details(graph, node['id'], agent_id, evidence)
                count += 1
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
        return [{'agent': agent, 'score': len(evidence), 'verified_tasks': len(evidence),
                 'evidence': sorted(evidence, key=lambda item: -item['verified_at']),
                 **economy.summary(evidence)}
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
                                 'completion': evidence['completion']}})
    graph.upsert_edge({'source': decision, 'target': ident, 'rel': 'verifies'})
    for quality in award['quality']:
        key = ident + ':quality:' + quality['kind']
        graph.upsert_node({'id': key, 'kind': 'work_quality_review', 'label': quality['kind'],
                           'attrs': {**quality, 'reviewer': evidence['verifier']}})
        graph.upsert_edge({'source': key, 'target': node, 'rel': 'earns_bonus'})
        graph.upsert_edge({'source': key, 'target': ident, 'rel': 'supported_by'})
    for field, kind, relation in (('review', 'work_rating', 'assesses'),
                                   ('usage', 'work_usage', 'measures')):
        if evidence.get(field) is not None:
            key = ident + ':' + field
            graph.upsert_node({'id': key, 'kind': kind, 'label': str(evidence['task']),
                               'attrs': {**evidence[field], 'reviewer': evidence['verifier']}})
            graph.upsert_edge({'source': key, 'target': ident, 'rel': relation})
