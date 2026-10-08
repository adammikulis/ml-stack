"""The per-project enforcement mode: open by default, strict restores the delegated-authority checks."""

import json

import pytest
from taskboard_kit import accepted, board as _board_fixture, proposed

from ml_stack.workspace import enforcement, enforcement_check, task_scope
from ml_stack.workspace.identity import Denied

board = _board_fixture
pytestmark = pytest.mark.redteam


def mode(kit, new):
    return enforcement.set_mode(kit.ws, kit.parent, '', new)


def task(kit, key, **extra):
    return kit.board.create(kit.parent, {**kit.spec, 'source_key': key, **extra})


def test_default_mode_is_open(board):
    assert enforcement.current(board.ws, '') == {'project': '', 'mode': 'open'}
    assert enforcement.mode(board.ws, {'key': 'anything'}) == 'open'


def test_unassigned_task_by_an_agent_is_claimable_by_peers_only_when_open(board):
    peer = board.ws.auth(board.agent('peer'))
    open_task = task(board, 'unassigned')
    assert task_scope.eligible(board.ws, peer, open_task)
    mode(board, 'strict')
    assert not task_scope.eligible(board.ws, peer, open_task)
    assert task_scope.eligible(board.ws, board.ws.auth(board.child), open_task)
    mode(board, 'open')
    assert task_scope.eligible(board.ws, peer, open_task)


def test_assignee_needs_the_task_project_only_when_strict(board):
    assert task(board, 'open-assignee', assignees=[board.worker_id])['assignees'] == [board.worker_id]
    mode(board, 'strict')
    with pytest.raises(Denied):
        task(board, 'strict-assignee', assignees=[board.worker_id])
    assert task(board, 'strict-plain')['assignees'] == []


def test_new_reviews_need_delegated_authority_only_when_strict(board):
    proposed(board)
    peer = board.agent('peer')
    mode(board, 'strict')
    with pytest.raises(Denied):
        board.board.review(peer, board.task['id'], accepted())
    mode(board, 'open')
    assert board.board.review(peer, board.task['id'], accepted())['accepted']


def test_promotion_keeps_in_flight_work_running(board):
    board.board.claim(board.child, board.task['id'], board.allocation['allocation_id'])
    mode(board, 'strict')
    board.board.heartbeat(board.child, board.task['id'])
    board.board.submit(board.child, board.task['id'], {
        'artifacts': {'replay.json': 'a' * 64}, 'checks': [{'name': 'Worker claims tests', 'passed': True}],
        'summary': 'proposed', 'provenance': {'commit': 'abc123', 'environment': 'native-python',
                                              'model': 'qwen', 'runtime': 'Agents SDK'}})
    assert board.board.get(board.parent, board.task['id'])['state'] == 'review'


def test_accepted_review_stays_valid_after_promotion(board):
    proposed(board)
    board.board.review(board.agent('peer'), board.task['id'], accepted())
    mode(board, 'strict')
    assert board.board.assert_reviewer(board.agent('peer2'), board.task['id'])['id'] == board.task['id']


def test_check_lists_what_strict_would_strand_without_changing_the_mode(board):
    task(board, 'peer-claimable')
    task(board, 'designated', assignees=[board.worker_id])
    report = enforcement_check.check(board.ws, '')
    kinds = {(row['kind'], row['task']) for row in report['stranded']}
    assert ('unassigned', board.task['id']) in kinds
    assert {row['kind'] for row in report['stranded']} == {'unassigned', 'designated'}
    assert report['mode'] == 'open' and not report['clear']
    assert enforcement.current(board.ws, '')['mode'] == 'open'


def test_accepted_review_by_a_peer_is_reported(board):
    proposed(board)
    board.board.review(board.agent('peer'), board.task['id'], accepted())
    assert board.board.get(board.parent, board.task['id'])['state'] == 'accepted'
    stranded = enforcement_check.check(board.ws, '')['stranded']
    assert [row['kind'] for row in stranded if row['task'] == board.task['id']] == ['review']


def test_lead_promotes_and_demotes_and_every_change_is_audited(board):
    first = enforcement.set_mode(board.ws, board.parent, '', 'strict')
    assert first['from'] == 'open' and enforcement.mode(board.ws, {}) == 'strict'
    enforcement.set_mode(board.ws, board.parent, '', 'open')
    changes = [row for row in board.ws.audit_log.rows() if row['event'] == 'enforcement.set']
    assert [(row['who'], row['project'], row['from'], row['to']) for row in changes] == [
        ('lead', '', 'open', 'strict'), ('lead', '', 'strict', 'open')]
    assert all(row['ts'] for row in changes)


def test_a_helper_child_cannot_set_the_mode(board):
    with pytest.raises(Denied):
        enforcement.set_mode(board.ws, board.child, '', 'strict')
    assert enforcement.mode(board.ws, {}) == 'open'
    assert not [row for row in board.ws.audit_log.rows() if row['event'] == 'enforcement.set']


def test_the_modes_are_per_project(board):
    enforcement.set_mode(board.ws, board.parent, 'alpha', 'strict')
    assert enforcement.mode(board.ws, {'key': 'alpha'}) == 'strict'
    assert enforcement.mode(board.ws, {'key': 'beta'}) == 'open'
    assert enforcement.mode(board.ws, {}) == 'open'


def test_cli_promote_demote_check_and_whoami(board):
    from workspace_kit import cli
    token = board.parent
    run = cli(board.ws.base, token, 'enforcement', 'check', '--project', 'p', '--json')
    assert run.returncode == 0, run.stderr
    shown = json.loads(run.stdout)
    assert shown['mode'] == 'open' and shown['clear']
    promoted = json.loads(cli(board.ws.base, token, 'enforcement', 'promote', '--project', 'p', '--json').stdout)
    assert promoted['mode'] == 'strict' and promoted['from'] == 'open' and promoted['preflight']['clear']
    assert json.loads(cli(board.ws.base, token, 'enforcement', '--project', 'p', '--json').stdout)['mode'] == 'strict'
    demoted = json.loads(cli(board.ws.base, token, 'enforcement', 'demote', '--project', 'p', '--json').stdout)
    assert demoted['mode'] == 'open' and demoted['from'] == 'strict'
    assert json.loads(cli(board.ws.base, token, 'whoami', '--json').stdout)['enforcement'] == 'open'
