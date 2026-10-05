"""Canonical graph tasks enforce native allocations, ownership and independent review."""

import concurrent.futures
from pathlib import Path

import pytest
from taskboard_kit import accepted, board, proposed

from ml_stack.graph.store import GraphStore
from ml_stack.workspace import resource_allocations, tokens
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.taskboard import TaskBoard, record, save

__all__ = ['board']


def test_source_dedup_graph_artifacts_and_reopened_accepted_outcome(board):
    task = board.task
    assert board.board.create(board.parent, board.spec)['id'] == task['id']
    proposal = proposed(board)
    review = board.board.review(board.parent, task['id'], accepted())
    result = TaskBoard(board.ws).get(board.child, task['id'])
    assert result['state'] == 'accepted' and result['proposal']['proposal_hash'] == proposal['proposal_hash']
    assert result['review']['proposal_id'] == proposal['id']
    assert result['review']['review_hash'] == review['review_hash']
    with GraphStore(board.base / 'coordination.db') as graph:
        assert {'task', 'agent', 'device', 'allocation', 'lease', 'proposal', 'artifact', 'review'} <= {n['kind'] for n in graph.nodes()}
        assert {'depends-on', 'produced-artifact'} & {edge['rel'] for edge in graph.edges()} == {'produced-artifact'}
    metrics = board.board.list(board.child)['metrics']
    assert metrics['verified_outcomes'] == 1 and metrics['accepted_artifacts'] == 1
    with pytest.raises(ValueError, match='source key'):
        board.board.create(board.parent, {**board.spec, 'title': 'Changed request'})


@pytest.mark.redteam
def test_duplicate_claim_and_released_resource_never_grant_task_ownership(board):
    def claim(_):
        try:
            return board.board.claim(board.child, board.task['id'], board.allocation['allocation_id'])
        except Denied:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        leases = list(pool.map(claim, range(2)))
    assert sum(lease is not None for lease in leases) == 1
    board.status['servers'][0]['holders'] = []
    with pytest.raises(Denied, match='active managed'):
        board.board.checkpoint(board.child, board.task['id'], {'summary': 'Cannot bypass released resource'})
    assert board.board.get(board.child, board.task['id'])['checkpoints'] == []


@pytest.mark.redteam
@pytest.mark.parametrize('reviewer', ['self', 'unrelated', 'missing'])
def test_only_actual_person_or_creator_parent_can_accept_proposal(board, reviewer):
    proposed(board)
    token = board.child if reviewer == 'self' else board.agent('other') if reviewer == 'unrelated' else ''
    with pytest.raises(Denied):
        board.board.review(token, board.task['id'], accepted())
    assert board.board.get(board.parent, board.task['id'])['state'] == 'review'
    assert board.board.list(board.parent)['metrics']['verified_outcomes'] == 0


@pytest.mark.redteam
def test_worker_claims_do_not_satisfy_acceptance_and_mutated_artifacts_refuse_review(board):
    proposal = proposed(board)
    with pytest.raises(ValueError, match='independent passing'):
        board.board.review(board.parent, board.task['id'],
                           {'accepted': True, 'reason': 'Reuse worker claim', 'checks': proposal['checks']})
    with GraphStore(board.base / 'coordination.db') as graph:
        value = record(graph, proposal['id'], 'proposal')
        value['artifacts']['replay.json'] = 'b' * 64
        save(graph, 'proposal', value)
    with pytest.raises(ValueError, match='proposal hash'):
        board.board.review(board.parent, board.task['id'], accepted())
    assert board.board.get(board.parent, board.task['id'])['review'] is None


def test_dependencies_checkpoint_budget_and_infrastructure_outcome(board):
    dependent = board.board.create(board.parent, {**board.spec, 'source_key': '', 'deps': [board.task['id']]})
    from ml_stack.workspace.resource_allocations import assign

    board.prepare(dependent['id'])
    allocation = assign(board.ws, board.parent, board.worker_id, dependent['id'], 'native-grant')
    with pytest.raises(Denied, match='dependencies'):
        board.board.claim(board.child, dependent['id'], allocation['allocation_id'])
    proposed(board)
    board.board.review(board.parent, board.task['id'], accepted())
    with pytest.raises(Denied, match='dependencies'):
        board.board.claim(board.child, dependent['id'], allocation['allocation_id'])
    dependent = board.board.create(board.parent, {**board.spec, 'source_key': 'independent-infrastructure'})
    allocation = assign(board.ws, board.parent, board.worker_id, dependent['id'], 'native-grant')
    board.board.claim(board.child, dependent['id'], allocation['allocation_id'])
    checkpoint = board.board.checkpoint(board.child, dependent['id'], {'summary': 'Replay started', 'commit': 'abc123'})
    assert board.board.get(board.parent, dependent['id'])['checkpoints'][0]['id'] == checkpoint['id']
    board.board.submit(board.child, dependent['id'], {'artifacts': {'trace': 'a' * 64},
                                                    'checks': [{'name': 'Replay passes', 'passed': False}]})
    outcome = board.board.review(board.owner, dependent['id'], {'accepted': False, 'outcome': 'blocked_infrastructure',
                                'reason': 'Native simulator unavailable', 'checks': [{'name': 'Replay passes', 'passed': False}]})
    assert outcome['outcome'] == 'blocked_infrastructure'
    assert board.board.get(board.parent, dependent['id'])['failures'] == 0


@pytest.mark.redteam
def test_expired_lease_and_missing_capability_refuse_execution(board):
    from ml_stack.workspace.resource_allocations import assign

    task = board.board.create(board.parent, {**board.spec, 'source_key': '', 'capabilities': ['vision']})
    board.prepare(task['id'])
    allocation = assign(board.ws, board.parent, board.worker_id, task['id'], 'native-grant')
    with pytest.raises(Denied, match='capability'):
        board.board.claim(board.child, task['id'], allocation['allocation_id'])
    assert board.board.get(board.parent, task['id'])['state'] == 'queued'
    board.board.claim(board.child, board.task['id'], board.allocation['allocation_id'])
    board.now[0] += 121
    with pytest.raises(Denied, match='lease expired'):
        board.board.heartbeat(board.child, board.task['id'])
    assert board.board.get(board.parent, board.task['id'])['lease']['deadline'] < board.now[0]


@pytest.mark.redteam
def test_changed_task_specification_cannot_be_claimed(board):
    with GraphStore(board.base / 'coordination.db') as graph:
        task = record(graph, board.task['id'], 'task')
        task['acceptance'] = ['Skip native replay']
        save(graph, 'task', task)
    with pytest.raises(ValueError, match='specification changed'):
        board.board.claim(board.child, board.task['id'], board.allocation['allocation_id'])


def test_coordinator_identity_survives_graph_copy_to_new_root(board, tmp_path):
    import shutil

    from ml_stack.workspace import Workspace
    from ml_stack.workspace.coordination import workspace_id

    other = tmp_path / 'replica'
    other.mkdir()
    shutil.copy2(board.base / 'coordination.db', other / 'coordination.db')
    assert workspace_id(Workspace(other)) == board.board.workspace_id
    assert board.board.get(board.parent, board.task['id'])['workspace'] == board.board.workspace_id


def test_owned_blockage_records_elapsed_time_without_verified_progress(board):
    board.board.claim(board.child, board.task['id'], board.allocation['allocation_id'])
    board.board.block(board.child, board.task['id'], 'Waiting for native simulator')
    board.now[0] += 30
    metrics = board.board.list(board.parent)['metrics']
    assert metrics['blocked_seconds'] == 30
    assert metrics['verified_outcomes'] == 0 and metrics['accepted_artifacts'] == 0
    assert board.board.get(board.parent, board.task['id'])['lease']['active'] is False


@pytest.mark.redteam
def test_designated_peer_requires_existing_project_grant_and_cannot_self_review(board):
    peer = board.agent('independent-reviewer')
    owner = board.ws.auth(board.owner)
    project = {'root': '/approved/project'}
    board.ws.registry.set_project(owner, board.worker_id, project)
    spec = {**board.spec, 'source_key': 'peer-review', 'project': project,
            'reviewers': ['independent-reviewer']}
    with pytest.raises(Denied, match='project grants'):
        board.board.create(board.owner, spec)
    board.ws.registry.set_project(owner, 'independent-reviewer', project)
    board.task = board.board.create(board.owner, spec)
    board.prepare(board.task['id'])
    board.allocation = resource_allocations.assign(board.ws, board.parent, board.worker_id,
                                                   board.task['id'], 'native-grant')
    proposed(board)
    board.board.review(peer, board.task['id'], accepted())
    assert board.board.assert_reviewer(peer, board.task['id'])['state'] == 'accepted'
    board.ws.registry.set_project(owner, 'independent-reviewer', {'root': '/elsewhere'})
    with pytest.raises(Denied):
        board.board.assert_reviewer(peer, board.task['id'])


@pytest.mark.redteam
def test_expired_claim_requires_bounded_independent_recovery_and_blocked_explicit_resume(board):
    ident = board.task['id']
    board.board.claim(board.child, ident, board.allocation['allocation_id'])
    with pytest.raises(Denied, match='live task lease'):
        board.board.recover(board.owner, ident, 'Attempt premature takeover')
    board.now[0] += 121
    with pytest.raises(Denied):
        board.board.claim(board.child, ident, board.allocation['allocation_id'])
    board.board.recover(board.owner, ident, 'Previous worker heartbeat expired')
    board.board.claim(board.child, ident, board.allocation['allocation_id'])
    board.board.block(board.child, ident, 'Approval pending')
    with pytest.raises(Denied):
        board.board.claim(board.child, ident, board.allocation['allocation_id'])
    with pytest.raises(Denied):
        board.board.resume(board.child, ident, 'Self approved')
    board.now[0] += 1
    board.board.resume(board.owner, ident, 'Person approved continuation')
    board.board.claim(board.child, ident, board.allocation['allocation_id'])
    detail = board.board.get(board.owner, ident)
    assert [row['transition'] for row in detail['checkpoints']] == ['recover', 'resume']
    assert detail['failures'] == 1


@pytest.mark.redteam
def test_designated_reviewer_without_read_capability_cannot_grade_task(board):
    reviewer = board.ws.delegate(board.parent, 'no-read-reviewer', can=('send',))
    token = tokens.read_file(Path(reviewer['token_file']))
    project = {'root': '/approved/project'}
    board.ws.registry.set_project(board.ws.auth(board.owner), board.worker_id, project)
    board.ws.registry.set_project(board.ws.auth(board.owner), reviewer['id'], project)
    board.task = board.board.create(board.owner, {**board.spec, 'source_key': 'read-grant-check',
                                    'project': project, 'reviewers': [reviewer['id']]})
    board.prepare(board.task['id'])
    board.allocation = resource_allocations.assign(board.ws, board.parent, board.worker_id,
                                                   board.task['id'], 'native-grant')
    proposed(board)
    with pytest.raises(Denied, match='right to read'):
        board.board.review(token, board.task['id'], accepted())
