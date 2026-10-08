"""Canonical worker eligibility follows registry parentage, assignment and project membership."""

from ml_stack.workspace import enforcement
from ml_stack.workspace.identity import HUMAN


def eligible(ws, who, task):
    """Return whether an authenticated worker may execute this task."""
    own = ws.registry.info(who.id)
    inherited = ws.registry.info(who.parent).get('project') if who.parent else None
    if task['project'] and task['project'] not in (own.get('project'), inherited):
        return False
    if enforcement.mode(ws, task['project']) == 'strict':
        creator = ws.registry.info(task['created_by'])
        delegated = (who.id in task['assignees'] and bool(task['project'])
                     and own.get('project') == task['project'])
        return who.role != HUMAN and (delegated or who.parent == task['created_by']
                                      or (creator['role'] == HUMAN and not task['assignees']))
    delegated = who.id in task['assignees'] and (not task['project'] or own.get('project') == task['project'])
    return who.role != HUMAN and (delegated or who.parent == task['created_by'] or not task['assignees'])
