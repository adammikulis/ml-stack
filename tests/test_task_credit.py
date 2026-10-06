"""Credits require canonical independent review evidence and stable coordinator scope."""

import hashlib
import shutil
from dataclasses import replace

import pytest
from taskboard_kit import accepted, board as _board_fixture, proposed

from ml_stack.graph.store import GraphStore
from ml_stack.memory import vault
from ml_stack.reputation.sealed import SealedGraph
from ml_stack.reputation.work import WorkLedger
from ml_stack.workspace import (
    coordination,
    device_agent,
    localagent,
    task_credit,
    tokens,
    work_reputation,
)
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.service import Workspace

board = _board_fixture


@pytest.fixture
def ledger(tmp_path):
    held = WorkLedger(SealedGraph(tmp_path / 'credit.enc',
                      keys=vault.PassphraseKeys(lambda _: 'isolated canonical credit key')))
    yield held
    held.sealed.close()


def completed_review(board, token, ident, decision):
    result = board.board.review(token, ident, decision)
    if result['accepted']:
        with GraphStore(board.base / 'coordination.db') as graph:
            node = next(node for node in graph.nodes('task') if node['id'] == ident)
            graph.upsert_node({**node, 'attrs': {**node['attrs'], 'state': 'completed'}})
    return result


def reviewed(board, decision=None):
    proposal = proposed(board)
    result = completed_review(board, board.parent, board.task['id'], decision or accepted())
    return proposal, result


def test_canonical_review_earns_once_from_independent_checks_not_worker_claims(board, ledger):
    proposal, review = reviewed(board, {**accepted(),
        'quality': [{'kind': 'validated', 'reason': 'Reviewer reproduced native replay.',
                     'checks': ['Replay passes'], 'artifacts': ['replay.json']}],
        'review': {'quality': 90, 'reliability': 100, 'reason': 'Independent inspection passed.'},
        'verified_usage': {'source': 'Independent task review', 'tokens_in': None,
                           'tokens_out': 24, 'wall_seconds': 2.0}})
    first = task_credit.verify_task(board.ws, board.parent, board.task['id'], ledger=ledger)
    again = task_credit.verify_task(board.ws, board.parent, board.task['id'], ledger=ledger)
    assert first['credited'] and not again['credited'] and first['id'] == again['id']
    assert first['checks'] == review['checks'] and first['checks'] != proposal['checks']
    assert first['proposal_hash'] == proposal['proposal_hash']
    assert first['completion_hash'] == review['review_hash']
    row = work_reputation.standings(board.ws, board.child, ledger=ledger)['own']
    assert row['economy']['balance'] == 15 and row['economy']['spent'] == 0
    assert row['work_reputation']['reliability'] == 66.67
    assert row['work_reputation']['reliability_samples'] == 1
    assert row['work_reputation']['quality_samples'] == 1
    assert row['economy']['usage']['tokens_out'] == 24
    assert first['provenance']['model'] == 'qwen'
    graph = ledger.sealed.graph()
    assert len(graph.nodes('work_contribution')) == 1 and len(graph.nodes('work_award')) == 1
    assert len(graph.nodes('work_rating')) == 1 and len(graph.nodes('work_usage')) == 1


@pytest.mark.redteam
def test_worker_and_unrelated_verifier_cannot_award_canonical_completion(board, ledger):
    reviewed(board)
    for token in (board.child, board.agent('other'), ''):
        with pytest.raises(Denied):
            task_credit.verify_task(board.ws, token, board.task['id'], ledger=ledger)
    assert ledger.standings(work_reputation.scope(board.ws)) == []


@pytest.mark.redteam
@pytest.mark.parametrize('kind, field, value', [
    ('proposal', 'artifacts', {'forged.json': 'b' * 64}),
    ('review', 'checks', [{'name': 'Forged acceptance', 'passed': True}]),
    ('review', 'proposal_hash', 'c' * 64),
    ('task', 'acceptance', ['Rewritten after review']),
    ('task', 'workspace', 'workspace:' + '0' * 32),
])
def test_review_and_proposal_tampering_cannot_mint_credits(board, ledger, kind, field, value):
    reviewed(board)
    with GraphStore(board.base / 'coordination.db') as graph:
        node = graph.nodes(kind)[0]
        graph.upsert_node({**node, 'attrs': {**node['attrs'], field: value}})
    with pytest.raises((Denied, ValueError)):
        task_credit.verify_task(board.ws, board.parent, board.task['id'], ledger=ledger)
    assert ledger.standings(work_reputation.scope(board.ws)) == []


@pytest.mark.parametrize('outcome, expected', [('rejected', 33.33), ('blocked_infrastructure', 50)])
def test_rejected_and_infrastructure_reviews_award_nothing_and_rate_fairly(board, ledger, outcome, expected):
    reviewed(board, {'accepted': False, 'outcome': outcome, 'reason': 'Independent rejected or blocked outcome.',
                     'checks': [{'name': 'Replay passes', 'passed': False}]})
    result = task_credit.verify_task(board.ws, board.parent, board.task['id'], ledger=ledger)
    assert not result['credited']
    row = work_reputation.standings(board.ws, board.child, ledger=ledger)['own']
    assert row['economy']['balance'] == 0 and row['work_reputation']['reliability'] == expected
    assert row['work_reputation']['outcomes'][outcome] == 1
    assert row['work_reputation']['quality_samples'] == 0
    assert row['work_reputation']['reliability_samples'] == (1 if outcome == 'rejected' else 0)


def test_coordinator_id_survives_relocation_and_migration_preserves_award_identity(board, ledger, tmp_path):
    old = hashlib.sha256(str(board.base.resolve()).encode()).hexdigest()
    data = {'workspace': old, 'agent': board.worker_id, 'task': 171, 'completion': 172,
            'verifier': 'lead', 'verified_at': board.ws.clock(), 'task_hash': 'a' * 64,
            'completion_hash': 'b' * 64, 'checks': [{'name': 'Independent checks', 'passed': True}],
            'artifacts': {'result.json': 'c' * 64},
            'award': {'policy': 'completion-quality-v1', 'currency': 'work-credit',
                      'base': 10, 'quality_bonus': 0, 'total': 10, 'quality': []}}
    first = ledger.record(data)
    current = coordination.workspace_id(board.ws)
    assert ledger.migrate_namespace(old, current) == 1
    assert ledger.migrate_namespace(old, current) == 0
    retry = ledger.record({**data, 'workspace': current})
    assert not retry['credited'] and retry['id'] == first['id']
    copied = tmp_path / 'relocated'
    shutil.copytree(board.base, copied)
    other = Workspace(copied)
    assert coordination.workspace_id(other) == current
    assert work_reputation.scope(other) == work_reputation.scope(board.ws)
    assert ledger.standings(current)[0]['economy']['balance'] == 10
    third = Workspace(tmp_path / 'independent')
    assert coordination.workspace_id(third) != current


def test_concurrent_canonical_award_and_outcome_record_once(board, ledger):
    from concurrent.futures import ThreadPoolExecutor

    reviewed(board)
    def award(_):
        sealed = SealedGraph(ledger.sealed.path, keys=ledger.sealed.keys)
        try:
            return task_credit.verify_task(board.ws, board.parent, board.task['id'],
                                           ledger=WorkLedger(sealed))['credited']
        finally:
            sealed.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(award, range(2)))
    assert sorted(results) == [False, True]
    row = work_reputation.standings(board.ws, board.child, ledger=ledger)['own']
    assert row['economy']['balance'] == 10 and row['work_reputation']['outcomes']['accepted'] == 1


@pytest.mark.parametrize('model, balance', [('Qwen3.8-27B', 20), ('gemma', 10)])
def test_canonical_awards_follow_verified_model_family_not_device_or_worker_label(board, ledger, model, balance, monkeypatch):
    from pathlib import Path

    from ml_stack.workspace import (
        device_agent,
        localagent,
        resource_allocations as resources,
        tokens,
    )

    reviewed(board)
    task_credit.verify_task(board.ws, board.parent, board.task['id'], ledger=ledger)
    delegated = board.ws.delegate(board.parent, 'second-model')
    second, token = delegated['id'], tokens.read_file(Path(delegated['token_file']))
    localagent.save(board.ws, localagent.Agent('second-model', model, identity=second,
                                              profile='coding', project=str(board.source), pid=555, process_started=42))
    monkeypatch.setattr(device_agent, 'device_id', lambda: 'second-device')
    monkeypatch.setattr(resources, 'device_id', lambda: 'second-device')
    device_agent.bind_worker(board.ws, board.owner, 'second-model')
    board.status['servers'][0]['model'] = model
    board.status['servers'][0]['pid'] = 1001
    task = board.board.create(board.parent, {**board.spec, 'source_key': 'repo:demo/sim:issue:43'})
    board.prepare(task['id'], second)
    allocation = resources.assign(board.ws, board.parent, second, task['id'], 'native-grant')
    board.board.claim(token, task['id'], allocation['allocation_id'])
    board.board.submit(token, task['id'], {'artifacts': {'next.json': 'b' * 64},
                        'checks': [{'name': 'Worker claims tests', 'passed': True}],
                        'provenance': {'model': 'forged-qwen-label', 'runtime': 'Codex'}})
    completed_review(board, board.parent, task['id'], accepted())
    task_credit.verify_task(board.ws, board.parent, task['id'], ledger=ledger)
    row = work_reputation.standings(board.ws, token, ledger=ledger)['own']
    members = balance // 10
    assert row['economy']['balance'] == balance and row['verified_tasks'] == members
    assert len(row['members']) == members and second in row['members']
    assert model in {item['provenance']['model'] for item in row['evidence']}
    assert row['account_type'] == 'model_family'
    team = work_reputation.standings(board.ws, board.parent, ledger=ledger)['team']
    qwen = next(item for item in team if item['family_id'] == 'qwen')
    assert qwen['economy']['balance'] == (20 if members == 2 else 10)
    if members == 2:
        assert set(qwen['devices']) == {'1234567890abcdef', 'second-device'}
    assert sum(item['economy']['balance'] for item in team) == 20
    assert len(ledger.sealed.graph().nodes('work_award')) == 2


@pytest.mark.redteam
def test_designated_peer_review_awards_with_live_person_grant_and_revocation_is_enforced(board, ledger):
    from ml_stack.workspace import resource_allocations as resources

    peer = board.agent('review-peer')
    project = {'root': '/approved/project'}
    board.ws.registry.set_project(board.ws.auth(board.owner), board.worker_id, project)
    board.ws.registry.set_project(board.ws.auth(board.owner), 'review-peer', project)
    task = board.board.create(board.owner, {**board.spec, 'project': project,
                              'reviewers': ['review-peer'], 'source_key': 'independent-peer'})
    board.prepare(task['id'])
    allocation = resources.assign(board.ws, board.parent, board.worker_id, task['id'], 'native-grant')
    board.board.claim(board.child, task['id'], allocation['allocation_id'])
    board.board.submit(board.child, task['id'], {'artifacts': {'reviewed.json': 'd' * 64},
                      'checks': [{'name': 'Worker claims acceptance', 'passed': True}]})
    completed_review(board, peer, task['id'], accepted())
    result = task_credit.verify_task(board.ws, peer, task['id'], ledger=ledger)
    assert result['credited'] and result['verifier'] == 'review-peer'
    board.ws.registry.set_project(board.ws.auth(board.owner), 'review-peer', {'root': '/different/project'})
    with pytest.raises(Denied):
        task_credit.verify_task(board.ws, peer, task['id'], ledger=ledger)
    assert work_reputation.standings(board.ws, board.child, ledger=ledger)['own']['economy']['balance'] == 10


@pytest.mark.redteam
def test_same_device_parent_cannot_review_through_another_worker_seat(board, ledger):
    localagent.save(board.ws, localagent.Agent('parent-worker', 'gemma', identity='lead',
                                               profile='coding'))
    tokens.store(board.base, 'lead', board.parent)
    device_agent.bind_worker(board.ws, board.owner, 'parent-worker')
    proposed(board)
    with pytest.raises(Denied, match='different enrolled device'):
        completed_review(board, board.parent, board.task['id'], accepted())
    completed_review(board, board.owner, board.task['id'], accepted())
    with pytest.raises(Denied, match='different enrolled device'):
        task_credit.verify_task(board.ws, board.parent, board.task['id'], ledger=ledger)
    assert task_credit.verify_task(board.ws, board.owner, board.task['id'], ledger=ledger)['credited']


def test_same_device_parent_still_recovers_expired_worker_lease(board):
    localagent.save(board.ws, localagent.Agent('parent-worker', 'gemma', identity='lead', profile='coding'))
    tokens.store(board.base, 'lead', board.parent)
    device_agent.bind_worker(board.ws, board.owner, 'parent-worker')
    board.board.claim(board.child, board.task['id'], board.allocation['allocation_id'])
    board.now[0] += 121
    recovered = board.board.recover(board.parent, board.task['id'], 'Recover expired owned worker.')
    assert recovered['state'] == 'queued'


def test_model_switch_binds_live_family_without_moving_historical_awards(board, ledger):
    from ml_stack.workspace import resource_allocations as resources

    board.board.claim(board.child, board.task['id'], board.allocation['allocation_id'])
    pending = work_reputation.standings(board.ws, board.child, ledger=ledger)['own']
    assert pending['family_id'] == 'qwen' and pending['economy']['balance'] == 0
    board.board.submit(board.child, board.task['id'], {'artifacts': {'replay.json': 'a' * 64},
                      'checks': [{'name': 'Worker claim', 'passed': True}]})
    completed_review(board, board.parent, board.task['id'], accepted())
    award = task_credit.verify_task(board.ws, board.parent, board.task['id'], ledger=ledger)
    board.now[0] += 30
    runner = replace(localagent.load(board.ws, 'native-worker'), model='gemma', model_name='gemma')
    localagent.save(board.ws, runner)
    board.status['servers'][0]['model'] = 'gemma'
    task = board.board.create(board.parent, {**board.spec, 'source_key': 'model-switch'})
    board.prepare(task['id'])
    allocation = resources.assign(board.ws, board.parent, board.worker_id, task['id'], 'native-grant')
    board.board.claim(board.child, task['id'], allocation['allocation_id'])
    result = work_reputation.standings(board.ws, board.child, ledger=ledger)
    assert result['own']['family_id'] == 'gemma' and result['own']['economy']['balance'] == 0
    qwen = next(row for row in result['team'] if row['family_id'] == 'qwen')
    assert qwen['economy']['balance'] == 10 and qwen['evidence'][0]['id'] == award['id']
    assert qwen['evidence'][0]['provenance']['model'] == 'qwen'
    assert len(ledger.sealed.graph().nodes('work_award')) == 1


@pytest.mark.redteam
def test_accepted_native_task_earns_nothing_before_integration_completes(board, ledger):
    proposed(board)
    board.board.review(board.parent, board.task['id'], accepted())
    with pytest.raises(Denied, match='completed canonical task'):
        task_credit.verify_task(board.ws, board.parent, board.task['id'], ledger=ledger)
    assert ledger.standings(work_reputation.scope(board.ws)) == []
