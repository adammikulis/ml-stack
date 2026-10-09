"""Read-only preflight listing what strict enforcement would stop for a project."""

from __future__ import annotations

from typing import Any

from ml_stack.workspace import enforcement
from ml_stack.workspace.identity import HUMAN
from ml_stack.workspace.taskboard import TaskBoard

LIVE = ('queued', 'working', 'review', 'blocked')


def check(ws, key: str) -> dict[str, Any]:
    """Tasks of project ``key`` that strict mode would strand, each with the reason."""
    board = TaskBoard(ws)
    with board._store() as graph:
        tasks = [node['attrs'] for node in graph.nodes('task')
                 if enforcement.key_of(node['attrs']['project']) == key]
        verifiers = {node['id']: node['attrs']['verifier'] for node in graph.nodes('review')}
    found: list[dict[str, str]] = []

    def note(task: dict[str, Any], kind: str, detail: str) -> None:
        found.append({'task': task['id'], 'title': task['title'], 'state': task['state'],
                      'kind': kind, 'detail': detail})

    for task in tasks:
        if task['state'] == 'queued' and not task['assignees'] \
                and ws.registry.role_of(task['created_by']) != HUMAN:
            note(task, 'unassigned', f'created by {task["created_by"]}; only its creator\'s own '
                 'workers could claim it')
        if task['state'] in LIVE:
            for identity in task['assignees'] + task['reviewers']:
                if not board._project_grant(identity, task, 'strict'):
                    note(task, 'designated', f'{identity} is not registered in the task project')
        if task['state'] == 'accepted' and task.get('review_id') in verifiers:
            verifier = verifiers[task['review_id']]
            if not board._qualifies(verifier, ws.registry.role_of(verifier), task, 'strict'):
                note(task, 'review', f'reviewer {verifier} would not qualify for a new review')
    return {'project': key, 'mode': enforcement.current(ws, key)['mode'], 'tasks': len(tasks),
            'stranded': found, 'clear': not found}
