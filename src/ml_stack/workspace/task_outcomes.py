"""Independent task reviews and idempotent outcome credit recording."""

from ml_stack.memory import vault
from ml_stack.workspace import task_credit
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.taskboard import TaskBoard


def review(ws, token, ident, decision):
    result = TaskBoard(ws).review(token, ident, decision)
    return {**result, 'credit': credit(ws, token, ident)}


def credit(ws, token, ident):
    try:
        result = task_credit.verify_task(ws, token, ident)
        total = result.get('award', {}).get('total', 0) if result.get('outcome') == 'accepted' else 0
        return {'state': 'recorded', 'credited': result['credited'], 'total': total,
                'evidence_id': result['id']}
    except Denied:
        return {'state': 'blocked', 'reason': 'This reviewer cannot record credit for this outcome.'}
    except (vault.KeyUnavailable, OSError, RuntimeError):
        return {'state': 'pending', 'reason': 'Credit recording is unavailable. The independent review remains saved.'}
