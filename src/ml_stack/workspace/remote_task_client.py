"""Typed task calls from parsed workspace commands."""

import json
import sys
import uuid

from ml_stack.workspace.identity import Denied

ACTIONS = frozenset({'tasks', 'task', 'task-create', 'task-claim', 'task-heartbeat',
                     'task-checkpoint', 'task-submit', 'task-review', 'task-credit'})
VALUES = frozenset({'task-checkpoint', 'task-submit', 'task-review'})


def command(remote, token: str, args):
    """Dispatch an authenticated typed task command."""
    action = args.cmd
    if action not in ACTIONS:
        raise Denied('this task operation is unavailable on the canonical board')
    payload = {}
    if action not in ('tasks', 'task-create'):
        payload['id'] = args.id
    if action == 'task-claim':
        payload['allocation_id'] = args.allocation_id
    if action == 'task-create' or action in VALUES:
        text = sys.stdin.read() if args.payload == '-' else args.payload
        payload['spec' if action == 'task-create' else 'value'] = json.loads(text)
    return remote.call('task.command', token, {
        'request_id': args.request_id or uuid.uuid4().hex,
        'action': action,
        'payload': payload,
    })
