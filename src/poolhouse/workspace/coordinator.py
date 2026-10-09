"""Agent-token coordinator RPC on the authenticated Fleet peer server."""

import json
import ssl

from poolhouse.fleet.onboard.web import Call
from poolhouse.workspace import (
    cli,
    coordinator_calls,
    coordinator_config,
    device_sessions,
    mesh_sync,
    onboard,
    tokens,
)
from poolhouse.workspace.chain import ChainBroken
from poolhouse.workspace.claims import Conflict
from poolhouse.workspace.coordination import workspace_id
from poolhouse.workspace.identity import Denied
from poolhouse.workspace.rates import RateLimited
from poolhouse.workspace.screen import Refused
from poolhouse.workspace.service import Workspace

PREFIX = '/workspace/v1/'
MAX_REQUEST = 32 * 1024


def answer(ws, call, *, device=None, projects=None):
    method, path, headers = call.method, call.path, call.headers
    body = call.body(MAX_REQUEST)
    if not call.secure:
        return 403, {'error': 'agent credentials require encrypted remote transport'}
    config = coordinator_config.load(ws.base)
    if config.get('mode') != 'host':
        return 503, {'error': 'this device is not the configured coordinator'}
    identity = workspace_id(ws)
    if identity != config['workspace']:
        return 409, {'error': 'coordinator workspace identity changed'}
    if method == 'GET' and path == PREFIX + 'info':
        return 200, {'workspace': identity, 'authority': 'coordinator', 'protocol': 1}
    if method != 'POST' or path not in (PREFIX + 'call', PREFIX + 'join', PREFIX + 'ensure'):
        return 405, {'error': 'unsupported coordination operation'}
    if len(body) > MAX_REQUEST:
        return 413, {'error': 'coordination request limit exceeded'}
    try:
        document = json.loads(body)
        if type(document) is not dict:
            raise ValueError('a coordination call is an object')
        if document.get('workspace') != identity:
            raise Denied('the requested coordinator workspace does not match')
        if path == PREFIX + 'ensure':
            if set(document) - {'device'} != {'workspace', 'name', 'model', 'harness', 'project'}:
                raise ValueError('agent registration carries identity labels and project')
            name, token = device_sessions.ensure(ws, device, projects, document,
                                                headers.get('X-Poolhouse-Workspace-Token', ''))
            return 200, {'workspace': identity, 'agent': name, 'token': token}
        if path == PREFIX + 'join':
            if set(document) != {'workspace', 'code', 'name', 'model', 'harness'} or any(
                    type(value) is not str for value in document.values()):
                raise ValueError('join carries an existing invitation and claimed model labels')
            name = onboard.join(ws, document['code'], document['name'],
                                claim=(document['model'], document['harness']))
            return 200, {'workspace': identity, 'agent': name, 'token': tokens.load(ws.base, name)}
        token = headers.get('X-Poolhouse-Workspace-Token', '')
        device_sessions.check(ws, token, device, projects)
        handlers = {name: handler for name, _help, _options, handler in cli.TABLE}
        result = coordinator_calls.execute(ws, token, document, cli.COMMANDS.parser(), handlers)
        if len(json.dumps(result, ensure_ascii=True).encode()) > coordinator_calls.MAX_OUTCOME:
            raise Denied('the coordination response exceeds its bound')
        return 200, {'workspace': identity, 'result': result}
    except (Denied, Refused):
        return 403, {'error': 'coordination authority or operation refused'}
    except RateLimited:
        return 429, {'error': 'coordination quota exceeded'}
    except Conflict as error:
        return 409, {'error': 'coordination claim conflict', 'owner': error.owner}
    except (ValueError, TypeError, KeyError, UnicodeError):
        return 400, {'error': 'invalid coordination request'}
    except (OSError, RuntimeError, ChainBroken):
        return 503, {'error': 'coordination storage is unavailable; preserve the request ID'}


def route(handler, body=None):
    if mesh_sync.route(handler, body):
        return True
    path = handler.path.split('?')[0]
    if path not in (PREFIX + 'info', PREFIX + 'call', PREFIX + 'join', PREFIX + 'ensure'):
        return False
    encrypted = isinstance(handler.connection, ssl.SSLSocket) or handler.client_address[0] in ('127.0.0.1', '::1')
    call = Call(handler.command, path, handler.headers,
                handler.client_address[0], encrypted, lambda _most: body or b'')
    status, result = answer(Workspace(), call, device=getattr(handler, '_workspace_device', None),
                            projects=getattr(handler, '_workspace_projects', None))
    handler._send(status, result)
    return True
