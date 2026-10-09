"""Structured task inspection and person-reviewed outcomes behind Fleet authorization."""

import json

from poolhouse.fleet.request_fields import object_body
from poolhouse.workspace import coordinator_config, limits, localroute, task_outcomes, tokens
from poolhouse.workspace.boardroute import Request, _checked
from poolhouse.workspace.identity import HUMAN, Denied
from poolhouse.workspace.service import Workspace
from poolhouse.workspace.taskboard import TaskBoard


def route(request, *, workspace=None, prefix="/ui/tasks", project=None, actor=None) -> bool:
    if request.path != prefix:
        return False
    configured = coordinator_config.load(limits.root()) if workspace is None else {}
    if configured.get('mode') == 'remote':
        request.send(409, {'error': 'Task authority is on the selected shared coordinator.',
                           'coordinator': configured['endpoint'], 'workspace': configured['workspace']})
        return True
    if request.method not in ('GET', 'POST'):
        request.send(405, {'error': 'tasks support GET and POST'})
        return True
    if request.method == 'POST':
        headers = {key.lower(): value for key, value in request.handler.headers.items()}
        refused = (_checked(request.method, headers, request.handler.server.server_address[1], writes=True)
                   if actor is not None else localroute._post_refusal(Request(request.method, request.path, headers,
                                           request.handler.server.server_address[1], True)))
        if refused:
            code, _extra, raw = refused
            request.send(code, json.loads(raw))
            return True
    try:
        ws = Workspace() if workspace is None else workspace
        token = actor if actor is not None else tokens.read_file(tokens.directory(ws.base) / tokens.OWNER_FILE)
        if ws.auth(token).role != HUMAN:
            raise Denied('task inspection requires the workspace person owner')
        board = TaskBoard(ws)
        if request.method == 'GET':
            ident = request.asked('id', '')
            result = board.get(token, ident) if ident else board.list(token)
        else:
            body = object_body(request)
            action = body.get('action', 'create')
            if action == 'create' and set(body) <= {'action', 'spec'}:
                spec = body.get('spec')
                if project is not None:
                    if not isinstance(spec, dict):
                        raise ValueError('a task specification is required')
                    scope = spec.get('project', {})
                    if not isinstance(scope, dict) or scope.get('key', project['key']) != project['key']:
                        raise Denied('the task belongs to another project')
                    spec = {**spec, 'project': project}
                result = board.create(token, spec)
            elif action == 'review' and set(body) == {'action', 'id', 'decision'}:
                result = task_outcomes.review(ws, token, body['id'], body['decision'])
            elif action == 'credit' and set(body) == {'action', 'id'}:
                result = {'credit': task_outcomes.credit(ws, token, body['id'])}
            elif action in ('resume', 'recover') and set(body) == {'action', 'id', 'reason'}:
                result = getattr(board, action)(token, body['id'], body['reason'])
            else:
                raise ValueError('tasks accept create, independent review, resume or recovery requests')
        request.send(200, result)
    except Denied:
        request.send(403, {'error': 'task authority is unavailable'})
    except (ValueError, UnicodeError) as error:
        request.send(400, {'error': str(error)})
    except (OSError, RuntimeError):
        request.send(503, {'error': 'task coordination is unavailable or locked'})
    return True
