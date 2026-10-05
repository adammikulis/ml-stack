"""Canonical worker eligibility follows registry parentage and person-set project grants."""

from ml_stack.workspace.identity import HUMAN


def eligible(ws, who, task):
    """Return whether an authenticated worker may execute this task."""
    creator = ws.registry.info(task['created_by'])
    delegated = (who.id in task['assignees'] and bool(task['project'])
                 and ws.registry.info(who.id).get('project') == task['project'])
    return who.role != HUMAN and (delegated or who.parent == task['created_by']
                                  or (creator['role'] == HUMAN and not task['assignees']))
