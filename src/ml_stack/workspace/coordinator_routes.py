"""Trusted device coordinator routing and visible connection health."""

import json

from ml_stack.fleet.session import parse_cookie
from ml_stack.http import ServerError
from ml_stack.workspace import (
    coordinator_client,
    coordinator_config,
    device_agent,
    limits,
    localroute,
    tokens,
)
from ml_stack.workspace.boardroute import Request
from ml_stack.workspace.chain import held
from ml_stack.workspace.coordination import workspace_id
from ml_stack.workspace.identity import HUMAN, Denied
from ml_stack.workspace.service import Workspace


def status(base):
    config = coordinator_config.load(base)
    visible = {key: value for key, value in config.items() if key != 'cert'}
    if not config:
        return {'mode': 'local', 'shared': False, 'connected': False}
    if config['mode'] == 'host':
        return {**visible, 'shared': True, 'connected': True}
    try:
        remote = coordinator_client.client(base)
        info = remote.peer._json('GET', '/workspace/v1/info')
        connected = info.get('workspace') == config['workspace'] and info.get('authority') == 'coordinator'
    except (ServerError, Denied, ValueError, OSError):
        connected = False
    return {**visible, 'shared': True, 'connected': connected}


def change(ws, token, document):
    with held(ws.base / 'coordinator-selection.lock'):
        return _change(ws, token, document)


def _change(ws, token, document):
    if type(document) is not dict:
        raise ValueError('coordinator selection is an object')
    local_agent = ws.auth(token).role != HUMAN
    if local_agent:
        device_agent.owned_local(ws, token)
    if document == {'action': 'host'}:
        if local_agent:
            raise Denied('only a person chooses what this device hosts')
        if coordinator_config.load(ws.base).get('mode') == 'remote':
            raise Denied('this device already follows a coordinator; it cannot create a second authority')
        return coordinator_config.save(ws.base, {'mode': 'host', 'workspace': workspace_id(ws)})
    if set(document) == {'action', 'name'} and document['action'] == 'connect' and type(document['name']) is str:
        if local_agent and coordinator_config.load(ws.base):
            raise Denied('an existing workspace authority cannot be replaced by agent selection')
        return coordinator_client._connect(ws.base, document['name'], replace=not local_agent)
    raise ValueError('choose host or an advertised coordinator name')


def route(request):
    if request.path != '/ui/coordination':
        return False
    base = limits.root()
    try:
        if request.method == 'GET':
            if request.asked('discover') == '1':
                result = {'coordinators': [{'name': peer.name, 'endpoint': peer.base_url, **info}
                                           for peer, info in coordinator_client.discover()]}
            else:
                result = status(base)
        elif request.method == 'POST':
            if request.client_ip not in ('127.0.0.1', '::1') or request.header('Authorization'):
                raise Denied('coordinator selection is made in this device browser')
            session = request.ui.sessions.get(parse_cookie(request.cookie))
            if session and session.who == 'token':
                raise Denied('an access-token session cannot select the coordinator')
            headers = {key.lower(): value for key, value in request.handler.headers.items()}
            refused = localroute._post_refusal(Request('POST', request.path, headers,
                                                     request.handler.server.server_address[1], True))
            if refused:
                code, _extra, raw = refused
                request.send(code, json.loads(raw))
                return True
            size = headers.get('content-length', '0')
            if not size.isascii() or not size.isdigit() or len(size) > 4 or int(size) > 1024:
                raise ValueError('coordinator selection exceeds its bound')
            ws = Workspace()
            change(ws, tokens.read_file(tokens.directory(base) / tokens.OWNER_FILE), request.body())
            result = status(base)
        else:
            request.send(405, {'error': 'coordinator selection supports GET and POST'})
            return True
        request.send(200, result)
    except Denied:
        request.send(403, {'error': 'coordinator selection or authority refused'})
    except (ValueError, ServerError, OSError):
        request.send(503, {'error': 'coordinator unavailable; pairing and workspace setup remain unchanged'})
    return True
