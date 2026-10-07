"""Task dashboard counts, authenticated assignments and evidence timestamps."""

from collections import Counter
from math import isfinite


def _time(value):
    return value if type(value) in (int, float) and isfinite(value) and value >= 0 else 0


def inspection(task, now):
    """Return task activity and dependency evidence without estimating worker claims."""
    lease = task.get('lease') or {}
    times = [_time(task.get(key)) for key in ('created_at', 'started_at', 'blocked_at')]
    times.extend(_time(lease.get(key)) for key in ('at', 'heartbeat_at'))
    for key in ('checkpoints', 'reviews', 'artifacts'):
        times.extend(_time(row.get('at')) for row in task.get(key, []))
    for key in ('proposal', 'review'):
        times.append(_time((task.get(key) or {}).get('at')))
    review = task.get('review') or {}
    checks = review.get('checks', [])
    return {'updated_at': max(times), 'age_seconds': max(0, now - _time(task.get('created_at'))),
            'worker_active': bool(task.get('state') == 'working' and lease.get('active')
                                  and _time(lease.get('deadline')) > now),
            'heartbeat_age_seconds': max(0, now - _time(lease['heartbeat_at']))
                                     if lease.get('heartbeat_at') is not None else None,
            'waiting_on': [dep for dep in task.get('dependencies', []) if dep['state'] != 'completed'],
            'verified_checks': {'passed': sum(check.get('passed') is True for check in checks),
                                'total': len(checks)},
            'eta_seconds': None, 'eta_reason': 'No verified numeric progress measurements.'}


def overview(tasks, now):
    """Summarize canonical task records at one server timestamp."""
    states = Counter(task['state'] for task in tasks)
    remaining = sum(task['state'] not in ('completed', 'rejected') for task in tasks)
    workers = []
    for task in tasks:
        if task['state'] != 'working' or not task.get('worker'):
            continue
        activity = inspection(task, now)
        workers.append({'id': task['worker'], 'task': task['id'], 'title': task['title'],
                        'device_id': task.get('device_id'), 'base_id': task.get('base_id'),
                        'active': activity['worker_active'],
                        'heartbeat_age_seconds': activity['heartbeat_age_seconds']})
    return {'as_of': now, 'total': len(tasks), 'remaining': remaining,
            'queued': states['queued'], 'running': states['working'], 'blocked': states['blocked'],
            'review': states['review'], 'awaiting_integration': states['accepted'],
            'done': states['completed'], 'failed': states['rejected'], 'workers': workers,
            'eta_seconds': None, 'eta_reason': 'No verified numeric progress measurements.'}
