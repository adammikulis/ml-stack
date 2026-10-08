"""Task overview derives counts and activity from evidence."""

from ml_stack.workspace.task_summary import inspection, overview


def test_overview_counts_unfinished_integration_and_expired_worker_leases():
    tasks = [{'id': str(index), 'title': state, 'state': state, 'created_at': 10}
             for index, state in enumerate(['queued', 'working', 'blocked', 'review', 'accepted', 'completed', 'rejected'])]
    tasks[1].update(worker='parent/worker', device_id='device-a', base_id='account-a',
                    lease={'active': True, 'heartbeat_at': 60, 'deadline': 90})
    value = overview(tasks, 100)
    assert {key: value[key] for key in ('total', 'remaining', 'queued', 'running', 'blocked',
                                      'review', 'awaiting_integration', 'done', 'failed')} == {
        'total': 7, 'remaining': 5, 'queued': 1, 'running': 1, 'blocked': 1,
        'review': 1, 'awaiting_integration': 1, 'done': 1, 'failed': 1}
    assert value['workers'] == [{'id': 'parent/worker', 'task': '1', 'title': 'working',
                                'device_id': 'device-a', 'base_id': 'account-a', 'active': False,
                                'heartbeat_age_seconds': 40}]
    assert value['eta_seconds'] is None


def test_worker_claimed_progress_cannot_create_verified_eta():
    task = {'state': 'working', 'created_at': 10, 'started_at': 20,
            'lease': {'active': True, 'heartbeat_at': 90, 'deadline': 120},
            'dependencies': [{'id': 'one', 'state': 'accepted'}, {'id': 'two', 'state': 'completed'}],
            'checkpoints': [{'at': 60, 'progress': '99 percent complete'},
                            {'at': 80, 'progress': 0.99}],
            'review': {'at': 70, 'checks': [{'passed': True}, {'passed': False}]}}
    value = inspection(task, 100)
    assert value['eta_seconds'] is None
    assert value['worker_active'] is True and value['heartbeat_age_seconds'] == 10
    assert value['updated_at'] == 90 and value['age_seconds'] == 90
    assert value['waiting_on'] == [{'id': 'one', 'state': 'accepted'}]
    assert value['verified_checks'] == {'passed': 1, 'total': 2}
