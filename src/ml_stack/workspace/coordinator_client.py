"""Fleet-enrolled shared coordination without a local-state fallback."""

import json
from urllib.parse import urlsplit
from uuid import uuid4

from ml_stack import http
from ml_stack.fleet import tls
from ml_stack.fleet.discovery import derive_token, load_cluster_key
from ml_stack.fleet.remote import Peer
from ml_stack.workspace import coordinator_config, tokens
from ml_stack.workspace.identity import Denied


class Remote:
    def __init__(self, config, peer):
        self.config, self.peer = config, peer

    def command(self, argv, token, *, request_id=''):
        payload = {'workspace': self.config['workspace'], 'request_id': request_id or uuid4().hex,
                   'argv': argv}
        try:
            _status, body, _headers = self.peer._request(
                'POST', '/workspace/v1/call', data=json.dumps(payload).encode(),
                headers={'Content-Type': 'application/json', 'X-ML-Stack-Workspace-Token': token})
        except http.ServerError as error:
            error.args = (f"Coordinator request {payload['request_id']}: {error}",)
            raise
        result = json.loads(body)
        if result.get('workspace') != self.config['workspace']:
            raise Denied('the responding coordinator workspace identity changed')
        return result['result']

    def join(self, base, code, name, model='', harness=''):
        result = self.peer._json('POST', '/workspace/v1/join',
                                 {'workspace': self.config['workspace'], 'code': code,
                                  'name': name, 'model': model, 'harness': harness})
        if result.get('workspace') != self.config['workspace']:
            raise Denied('the responding coordinator workspace identity changed')
        tokens.store(base, result['agent'], result['token'])
        return result['agent']


def client(base):
    config = coordinator_config.load(base)
    if config.get('mode') != 'remote':
        return None
    key = load_cluster_key()
    if key is None:
        raise Denied('join the coordinator Fleet cluster before connecting')
    if config.get('cert'):
        http.pin(urlsplit(config['endpoint']).netloc, tls.pinned_context(config['cert']))
    return Remote(config, Peer(config['endpoint'], derive_token(key), timeout=30))


def discover():
    candidates = []
    for peer in Peer.discover(timeout_s=2):
        try:
            info = peer._json('GET', '/workspace/v1/info')
        except http.ServerError:
            continue
        if info.get('authority') == 'coordinator' and info.get('protocol') == 1:
            candidates.append((peer, info))
    return candidates


def connect(base, name):
    candidates = [(peer, info) for peer, info in discover() if not name or peer.name == name]
    if len(candidates) != 1:
        raise Denied('select one advertised coordinator by its Fleet name; none or several matched')
    peer, info = candidates[0]
    coordinator_config.validate_endpoint(peer.base_url)
    return coordinator_config.save(base, {'mode': 'remote', 'workspace': info['workspace'],
                                         'name': peer.name, 'endpoint': peer.base_url,
                                         'cert': peer.beacon.cert if peer.beacon else ''})


def argv_for(args, declarations):
    argv = [args.cmd]
    for declaration in declarations:
        flags, options = declaration.flags, declaration.kwargs
        name = flags[-1].lstrip('-').replace('-', '_')
        if name in ('agent', 'token_file', 'json', 'request_id'):
            continue
        value = getattr(args, options.get('dest', name))
        if not flags[0].startswith('-'):
            argv.extend(str(item) for item in value) if isinstance(value, list) else argv.append(str(value))
        elif options.get('action') == 'store_true':
            if value:
                argv.append(flags[-1])
        elif value != options.get('default') and value is not None:
            argv.extend([flags[-1], str(value)])
    return argv
