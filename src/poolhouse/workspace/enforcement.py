"""The per-project task enforcement mode: `open` or `strict`, set by a lead or a person."""

from __future__ import annotations

from typing import Any

from poolhouse.graph.store import GraphStore
from poolhouse.workspace.chain import held
from poolhouse.workspace.identity import Denied

MODES = ('open', 'strict')
DEFAULT = 'open'


def key_of(project: dict[str, str] | None) -> str:
    """The project key a task or identity names; empty when it names none."""
    return (project or {}).get('key', '')


def current(ws, key: str) -> dict[str, Any]:
    """The enforcement record for project ``key``; `open` when none was set."""
    path = ws.base / 'enforcement.db'
    if path.exists():
        with held(ws.base / 'enforcement.lock', shared=True), GraphStore(path) as graph:
            found = next((node['attrs'] for node in graph.nodes('enforcement')
                          if node['id'] == f'enforcement:{key}'), None)
        if found:
            return found
    return {'project': key, 'mode': DEFAULT}


def mode(ws, project: dict[str, str] | None) -> str:
    """The enforcement mode that applies to a task or identity's ``project``."""
    return current(ws, key_of(project))['mode']


def set_mode(ws, token: str, key: str, new: str) -> dict[str, Any]:
    """Set project ``key`` to ``new`` in one step and audit it; a helper identity is refused."""
    who = ws.auth(token)
    if who.parent:
        ws.audit('auth.denied', who.id, reason='enforcement mode is set by a lead, not a helper')
        raise Denied('only a lead or a person sets the task enforcement mode, not a helper')
    if new not in MODES:
        raise ValueError(f'enforcement mode must be one of {", ".join(MODES)}')
    with held(ws.base / 'enforcement.lock'), GraphStore(ws.base / 'enforcement.db') as graph, graph.transaction():
        before = next((node['attrs']['mode'] for node in graph.nodes('enforcement')
                       if node['id'] == f'enforcement:{key}'), DEFAULT)
        record = {'id': f'enforcement:{key}', 'project': key, 'mode': new, 'set_by': who.id,
                  'at': ws.clock()}
        graph.upsert_node({'id': record['id'], 'kind': 'enforcement', 'label': key or 'workspace',
                           'attrs': record})
    ws.audit('enforcement.set', who.id, project=key, **{'from': before, 'to': new})
    return {**record, 'from': before}
