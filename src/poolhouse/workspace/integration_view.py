"""Bounded task integration inspection without execution credentials or private paths."""

import re

from poolhouse.sentinel.redaction import redact

FIELDS = ('id', 'task', 'state', 'owner', 'worker', 'review_id', 'review_hash',
          'proposal_id', 'proposal_hash', 'source_commit', 'commit', 'development',
          'policy', 'at', 'started', 'finished', 'blocking_owner')


def public(record: dict, private_root: str) -> dict:
    result = {key: record[key] for key in FIELDS if key in record}
    if record.get('reason'):
        reason = re.sub(re.escape(private_root) + r'[^\s\"\']*', '[workspace private path]',
                        str(record['reason']))
        if record.get('candidate'):
            reason = reason.replace(record['candidate'], '[integration candidate]')
        result['reason'] = redact(reason)[:2000]
    if record.get('checks'):
        result['checks'] = [{key: check[key] for key in ('passed', 'exit', 'output_hash') if key in check}
                            | {'command': check['command'][1:]}
                            for check in record['checks']]
    if record.get('blocking_claim'):
        claim = record['blocking_claim']
        result['blocking_claim'] = {key: claim[key] for key in ('kind', 'owner') if key in claim}
        if claim.get('kind') == 'branch':
            result['blocking_claim']['key'] = claim['key']
    return result


def inspection(children: dict, index: dict, review_id: str | None, private_root: str) -> dict:
    integrations = []
    for record in children['integration']:
        review = index.get(record['review_id'], {})
        integrations.append(public({**record, 'proposal_id': review.get('proposal_id')}, private_root))
    attempts = []
    for record in sorted(children['integration-attempt'], key=lambda row: row['started']):
        review = next((row for row in children['review'] if row['review_hash'] == record['review_hash']), {})
        binding = {'review_id': review.get('id'), 'proposal_id': review.get('proposal_id'),
                   'proposal_hash': review.get('proposal_hash')}
        attempt = public({**record, **binding}, private_root)
        if record.get('outcome'):
            attempt['outcome'] = public({**record['outcome'], **binding}, private_root)
        attempts.append(attempt)
    return {'integration': next((row for row in integrations if row['review_id'] == review_id), None),
            'integrations': integrations, 'integration_attempts': attempts,
            'integration_events': [public(row, private_root) for row in children['integration-event']]}
