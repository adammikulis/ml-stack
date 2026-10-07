"""Bounded execution-profile observations under an authenticated agent identity."""
from __future__ import annotations

import math
import os
import re
import uuid

from ml_stack.graph.columns import column
from ml_stack.graph.store import GraphStore
from ml_stack.platform import private_file
from ml_stack.windows_private import restrict
from ml_stack.workspace import tokens
from ml_stack.workspace.chain import held
from ml_stack.workspace.coordination import workspace_id
from ml_stack.workspace.identity import AGENT, Denied

TEXT = frozenset({'model', 'model_version', 'harness', 'harness_version', 'runtime_commit',
                  'runtime_version', 'python_version', 'requested_model', 'requested_effort',
                  'effective_effort', 'requested_service_tier', 'effective_service_tier',
                  'harness_agent_id', 'harness_agent_type'})
LIMITS = frozenset({'requested_context', 'effective_context', 'requested_output_tokens',
                    'effective_output_tokens', 'requested_turns', 'effective_turns',
                    'requested_tool_calls', 'effective_tool_calls', 'requested_model_calls',
                    'effective_model_calls', 'requested_wall_seconds', 'effective_wall_seconds',
                    'requested_resource_units', 'effective_resource_units'})
FIELDS = TEXT | LIMITS
SOURCES = frozenset({'verified', 'reported', 'inherited', 'unknown'})
OPAQUE = re.compile(r'[A-Za-z0-9_.:-]{1,128}')
HISTORY = 64


def _document(document):
    if type(document) is not dict or set(document) != {'session', 'event_id', 'event', 'fields'}:
        raise ValueError('execution observations carry session, event ID, event and fields only')
    for name in ('session', 'event_id', 'event'):
        if type(document[name]) is not str or not OPAQUE.fullmatch(document[name]):
            raise ValueError('execution observation identifiers must be bounded opaque values')
    values = document['fields']
    if type(values) is not dict or set(values) - FIELDS:
        raise ValueError('execution observation fields must use the metadata whitelist')
    for key, value in values.items():
        if value is None:
            continue
        if key in TEXT:
            if type(value) is not str or not value or len(value) > 256 or any(ord(c) < 32 for c in value):
                raise ValueError('execution text metadata must be bounded printable text')
            if key == 'runtime_commit' and not re.fullmatch(r'[a-f0-9]{40}', value):
                raise ValueError('runtime metadata requires a full source commit')
        elif type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 10**12:
            raise ValueError('execution limits must be finite independent nonnegative values')
    return document


def _path(ws):
    path = ws.base / 'coordination.db'
    for candidate in (path, ws.base / 'coordination.lock', ws.base):
        why = tokens.problem(candidate)
        if why.startswith('mode ') and candidate != ws.base:
            private_file(candidate)
        elif why not in ('', 'missing'):
            raise Denied(f'execution profile storage is not private: {why}')
    return path


def _rows(graph, actor, project):
    records = graph.query('MATCH (n:Node)-[e:Edge]->(a:Node) '
                          'WHERE n.kind=$kind AND e.rel=$rel AND a.id=$actor RETURN n.attrs AS attrs',
                          {'kind': 'execution-observation', 'rel': 'observed-for', 'actor': f'agent:{actor}'})
    rows = [column(record['attrs'], 'execution observation') for record in records]
    return [row for row in rows if row['actor'] == actor and row['project'] == project]


def _write(ws, who, document, sources):
    path = _path(ws)
    info = ws.registry.info(who.id)
    project = info.get('project', {}).get('key') or workspace_id(ws)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == 'nt':
        restrict(path.parent)
    with held(ws.base / 'coordination.lock'), GraphStore(path) as graph:
        restrict(path) if os.name == 'nt' else private_file(path)
        rows = _rows(graph, who.id, project)
        previous = next((row for row in rows if row['event_id'] == document['event_id']), None)
        if previous:
            if previous['document'] != document:
                raise ValueError('an observation event ID cannot name different metadata')
            return previous
        session = document['session']
        candidates = [row for row in rows if row['session'] == session]
        latest = max(candidates, key=lambda row: row['sequence'], default=None)
        fields = dict(latest['fields']) if latest else {key: {'value': None, 'source': 'unknown'} for key in sorted(FIELDS)}
        for key, value in document['fields'].items():
            fields[key] = {'value': value, 'source': sources[key] if value is not None else 'unknown'}
        row = {'id': f'observation:{uuid.uuid4().hex}', 'actor': who.id, 'parent': who.parent or None,
               'project': project, 'device': info.get('device') or None, 'session': session, 'event_id': document['event_id'],
               'event': document['event'], 'observed_at': ws.clock(), 'fields': fields,
               'sequence': max((item['sequence'] for item in rows), default=0) + 1, 'document': document}
        graph.upsert_node({'id': row['id'], 'kind': 'execution-observation', 'label': document['event'], 'attrs': row})
        if not graph.has(f'agent:{who.id}'):
            graph.upsert_node({'id': f'agent:{who.id}', 'kind': 'agent', 'label': who.id})
        graph.upsert_edge({'source': row['id'], 'target': f'agent:{who.id}', 'rel': 'observed-for'})
        if who.parent:
            if not graph.has(f'agent:{who.parent}'):
                graph.upsert_node({'id': f'agent:{who.parent}', 'kind': 'agent', 'label': who.parent})
            graph.upsert_edge({'source': f'agent:{who.id}', 'target': f'agent:{who.parent}', 'rel': 'delegated-by'})
        if len(rows) >= HISTORY:
            graph.drop([min(rows, key=lambda item: item['sequence'])['id']])
    ws.audit('execution.observe', who.id, observation=row['id'], project=project, session=session,
             event_name=document['event'])
    return row


def record(ws, token, document):
    """Record only the authenticated caller's reported execution metadata."""
    who = ws.auth(token)
    ws._may(who, 'send')
    if who.role != AGENT:
        raise Denied('execution observations belong to an authenticated agent')
    document = _document(document)
    return _write(ws, who, document, dict.fromkeys(document['fields'], 'reported'))


def read(ws, token):
    """Return this authenticated actor's bounded execution observation history."""
    who = ws.auth(token)
    ws._may(who, 'read')
    path = _path(ws)
    if not path.exists():
        return []
    project = ws.registry.info(who.id).get('project', {}).get('key') or workspace_id(ws)
    with held(ws.base / 'coordination.lock'), GraphStore(path) as graph:
        return sorted(_rows(graph, who.id, project), key=lambda row: row['sequence'])
