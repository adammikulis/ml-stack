"""Actor and execution views of reviewed outcomes and existing award references."""

from collections import defaultdict

from poolhouse.graph.store import GraphStore
from poolhouse.reputation import economy
from poolhouse.workspace.chain import held
from poolhouse.workspace.task_schema import SPEC_FIELDS, fingerprint


def receipt(ws, result):
    """Link an existing independently recorded award to its review."""
    ident = 'work-credit-reference:' + result['completion']
    attrs = {key: result[key] for key in
             ('agent', 'workspace', 'completion', 'proposal_id', 'proposal_hash', 'completion_hash', 'id')}
    attrs['award'] = result.get('award') if result.get('outcome') == 'accepted' else None
    with held(ws.base / 'coordination.lock'), GraphStore(ws.base / 'coordination.db') as graph:
        graph.upsert_node({'id': ident, 'kind': 'work-credit-reference',
                           'label': result['completion'], 'attrs': attrs})
        graph.upsert_edge({'source': ident, 'target': result['completion'], 'rel': 'records-credit'})


def _checked(row, field):
    return fingerprint({key: value for key, value in row.items() if key != field}) == row.get(field)


def _bound(task, proposal, review, workspace):
    return task.get('workspace') == workspace and _checked(review, 'review_hash') \
        and _checked(proposal, 'proposal_hash') \
        and fingerprint({key: task.get(key) for key in SPEC_FIELDS}) == task.get('spec_hash') \
        and proposal.get('task') == review.get('task') == task.get('id') \
        and proposal.get('worker') == review.get('worker') != review.get('verifier') \
        and proposal.get('provenance', {}).get('actor', proposal.get('worker')) == proposal.get('worker') \
        and review.get('proposal_hash') == proposal.get('proposal_hash')


def _outcomes(ws, workspace):
    with held(ws.base / 'coordination.lock'), GraphStore(ws.base / 'coordination.db') as graph:
        tasks = {node['id']: node['attrs'] for node in graph.nodes('task')}
        proposals = {node['id']: node['attrs'] for node in graph.nodes('proposal')}
        references = {node['attrs']['completion']: node['attrs'] for node in graph.nodes('work-credit-reference')}
        reviews = [node['attrs'] for node in graph.nodes('review')]
    result = []
    for review in reviews:
        task = tasks.get(review['task'], {})
        proposal = proposals.get(review['proposal_id'], {})
        if not _bound(task, proposal, review, workspace):
            continue
        if review['outcome'] == 'accepted' and task.get('state') != 'completed':
            continue
        provenance = proposal.get('provenance', {})
        row = {**review, 'agent': proposal['worker'], 'source': 'taskboard',
               'verified_at': review['reviewed_at'], 'provenance': provenance,
               'family_account': proposal.get('family_account'), 'award': None,
               'artifacts': proposal['artifacts'], 'task_limits': task['limits']}
        if measured := review.get('verified_usage'):
            row['usage'] = {'source': measured.get('source', 'Independent task review'),
                            **{key: measured.get(key) for key in ('tokens_in', 'tokens_out', 'wall_seconds')}}
        reference = references.get(review['id'])
        if reference and all(reference.get(key) == value for key, value in {
                'agent': proposal['worker'], 'workspace': workspace, 'completion': review['id'],
                'proposal_id': proposal['id'], 'completion_hash': review['review_hash'],
                'proposal_hash': proposal['proposal_hash']}.items()):
            row.update(award=reference['award'], evidence_id=reference['id'])
        result.append(row)
    return result


def _rows(groups, kind, offset):
    result = []
    for identity, items in sorted(groups.items()):
        evidence = [item for item in items if item['outcome'] == 'accepted' and item['award']]
        aliases = sorted({item['provenance']['alias'] for item in items if item['provenance'].get('alias')})
        ordered = sorted(items, key=lambda item: -item['verified_at'])
        execution = items[0]['provenance']
        details = {key: execution.get(key) for key in
                   ('model', 'runtime', 'model_source', 'runtime_source', 'artifact', 'broker_runtime')} if kind == 'model-runtime' else {}
        result.append({'id': identity, 'dimension': kind, 'aliases': aliases, **details,
                       'members': sorted({item['agent'] for item in items}),
                       'verified_tasks': len(evidence), **economy.summary(evidence, items),
                       'evidence': ordered[offset:offset + 20], 'evidence_held': max(0, len(items) - offset - 20)})
    return result


def views(ws, workspace, *, agent='', offset=0):
    """Derive reputation dimensions without recording scores or additional awards."""
    agents, models = defaultdict(list), defaultdict(list)
    for item in _outcomes(ws, workspace):
        if agent and item['agent'] != agent:
            continue
        agents[item['agent']].append(item)
        provenance = item['provenance']
        model, runtime = provenance.get('model'), provenance.get('runtime')
        artifact = provenance.get('artifact') or {}
        identity = fingerprint({'model': model, 'runtime': runtime,
                                'model_source': provenance.get('model_source', 'unknown'),
                                'artifact': artifact.get('id'),
                                'device': None if artifact.get('id') else provenance.get('device_id'),
                                'broker_runtime': provenance.get('broker_runtime')})
        models[identity].append(item)
    return {'agents': _rows(agents, 'agent', offset), 'models': _rows(models, 'model-runtime', offset),
            'award_policy': 'one-award-per-task; dimensions are overlapping views'}
