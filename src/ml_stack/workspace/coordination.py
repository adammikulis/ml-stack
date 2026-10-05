"""Stable coordinator identity in the maintained workspace graph."""

from uuid import uuid4

from ml_stack.graph.store import GraphStore
from ml_stack.workspace.chain import held


def workspace_id(ws) -> str:
    """Read or initialize the coordinator identity retained by graph replicas."""
    with held(ws.base / 'coordination.lock'), GraphStore(ws.base / 'coordination.db') as graph:
        nodes = graph.nodes('workspace')
        if len(nodes) > 1:
            raise ValueError('the coordination graph has multiple workspace identities')
        if nodes:
            return nodes[0]['id']
        ident = f'workspace:{uuid4().hex}'
        graph.upsert_node({'id': ident, 'kind': 'workspace', 'label': 'Workspace coordinator'})
        return ident
