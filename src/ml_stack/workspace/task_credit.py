"""Credit canonical task outcomes using their independently authenticated review."""

from ml_stack import activity
from ml_stack.reputation import economy
from ml_stack.workspace import task_schema, work_reputation
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.taskboard import TaskBoard


def verify_task(ws, token, task_id, *, ledger=None):
    """Award an accepted canonical task once; worker claims never supply verification checks."""
    who = ws.auth(token)
    board = TaskBoard(ws)
    board.assert_reviewer(token, task_id)
    task = board.get(token, task_id)
    if task['workspace'] != work_reputation.scope(ws):
        raise Denied('the task belongs to a different coordinator workspace')
    proposal, review = task.get('proposal'), task.get('review')
    if not proposal or not review:
        raise Denied('a canonical task needs an independently authenticated review')
    outcome = review['outcome']
    if outcome not in ('accepted', 'rejected', 'blocked_infrastructure') \
            or (outcome == 'accepted') != review['accepted']:
        raise Denied('the canonical review outcome is inconsistent')
    if outcome == 'accepted' and task['state'] != 'completed':
        raise Denied('only a completed canonical task earns completion credit')
    worker = proposal['worker']
    if who.id == worker or review['verifier'] != who.id:
        raise Denied('only the authenticated independent reviewer can award this task')
    if review['task'] != task['id'] or proposal['task'] != task['id'] \
            or review['worker'] != worker or review['proposal_id'] != proposal['id'] \
            or review['proposal_hash'] != proposal['proposal_hash']:
        raise Denied('the accepted review must bind this exact worker proposal')
    if task_schema.fingerprint({key: value for key, value in proposal.items() if key != 'proposal_hash'}) != proposal['proposal_hash']:
        raise Denied('the proposal content does not match its recorded hash')
    if task_schema.fingerprint({key: value for key, value in review.items() if key != 'review_hash'}) != review['review_hash']:
        raise Denied('the independent review content does not match its recorded hash')
    checks = task_schema.checks(review['checks'])
    if outcome == 'accepted' and not all(item['passed'] for item in checks):
        raise Denied('accepted task credit requires independently passing review checks')
    proof = {'checks': checks, 'artifacts': proposal['artifacts'],
             **{key: review[key] for key in ('quality', 'review') if key in review}}
    if review.get('verified_usage'):
        measured = review['verified_usage']
        proof['usage'] = {'source': measured.get('source', 'Independent task review'),
                          **{key: measured.get(key) for key in ('tokens_in', 'tokens_out', 'wall_seconds')}}
    if outcome != 'accepted' and proof.get('quality'):
        raise Denied('an unaccepted contribution cannot earn quality bonuses')
    evidence = {**proof, 'award': economy.assessment(proof)}
    if proposal.get('family_account'):
        evidence['family_account'] = proposal['family_account']
    if proposal.get('provenance'):
        evidence['provenance'] = proposal['provenance']
    data = {**evidence, 'agent': worker, 'verifier': who.id,
            'task': task['id'], 'completion': review['id'], 'outcome': outcome,
            'reason': review['reason'], 'workspace': work_reputation.scope(ws),
            'verified_at': review['reviewed_at'], 'task_hash': task['spec_hash'],
            'completion_hash': review['review_hash'],
            'proposal_id': proposal['id'], 'proposal_hash': proposal['proposal_hash'],
            'source': 'canonical-taskboard'}
    with work_reputation._store(ledger) as held:
        contribution = held.record_contribution({key: value for key, value in data.items() if key != 'award'})
        if outcome != 'accepted':
            return {**contribution, 'credited': False}
        result = held.record(data)
    if result['credited']:
        ws.audit('work.verified', who.id, agent=worker, task=task['id'], evidence=result['id'])
        activity.record('agent.work_verified', actor=worker, subject=task['title'], outcome='verified',
                        refs={'task': task['id'], 'evidence': result['id'], 'verifier': who.id},
                        meta={'credits': result['award']['total']})
    return result
