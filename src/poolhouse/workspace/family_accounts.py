"""Model-family accounts linked to actual task model resources."""

from poolhouse.client.families import GENERIC, for_model_id
from poolhouse.graph.store import GraphStore
from poolhouse.workspace.task_graph import link, save


def binding(workspace, resource):
    family = for_model_id(resource.get('model'))
    if family is GENERIC:
        return None
    return {'id': f'family:{workspace}:{family.name}', 'account': f'family-{family.name}',
            'family_id': family.name, 'label': {'qwen':'Qwen','gemma':'Gemma','gpt-oss':'GPT OSS'}[family.name],
            'workspace': workspace, 'source': 'verified-serving-resource'}


def bind_resource(graph, task, worker, resource, at):
    account = binding(task['workspace'], resource)
    if account is None:
        return None
    save(graph, 'model-family-account', account)
    graph.upsert_node({'id': f'agent:{worker}', 'kind': 'agent', 'label': worker})
    graph.upsert_edge({'source': f'agent:{worker}', 'target': account['id'], 'rel': 'model-family-worker',
                       'attrs': {'at': at, 'model': resource['model'], 'device_id': resource.get('device_id')}})
    link(graph, task['id'], account['id'], 'model-family-contribution')
    return account


def current_accounts(ws, identity):
    with GraphStore(ws.base / 'coordination.db') as graph:
        edges = [edge for edge in graph.edges('model-family-worker') if edge['source'] == f'agent:{identity}']
        targets = {edge['target']: edge.get('attrs', {}).get('at', 0) for edge in edges}
        return sorted([{**node['attrs'], 'last_bound': targets[node['id']]}
                       for node in graph.nodes('model-family-account') if node['id'] in targets],
                      key=lambda item: -item['last_bound'])
