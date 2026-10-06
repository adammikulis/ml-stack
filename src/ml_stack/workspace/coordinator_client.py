"""Fleet-enrolled shared coordination without a local-state fallback."""

import base64
import json
from urllib.parse import urlsplit
from uuid import uuid4

from ml_stack import home, http, macauth
from ml_stack.graph.store import GraphStore
from ml_stack.hub.peerbook import PeerBook
from ml_stack.fleet import tls
from ml_stack.fleet.onboard.requests import Devices
from ml_stack.fleet.discovery import load_cluster_key
from ml_stack.fleet.remote import Peer
from ml_stack.workspace import coordinator_config, tokens
from ml_stack.workspace.chain import held
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

    def ensure(self, base, name, model='', harness='', project=None):
        with held(base / 'remote-sessions.lock'), GraphStore(base / 'remote-sessions.db') as graph:
            key = 'remote-session:' + self.config['workspace'] + ':' + name
            rows = graph.nodes('remote-session')
            row = next((row['attrs'] for row in rows if row['id'] == key), {})
            agent = row.get('agent', name)
            credential = ''
            problem = tokens.problem(tokens.directory(base) / agent.replace('/', '~'))
            if problem and problem != 'missing':
                raise Denied('the agent credential file is unsafe')
            if not problem:
                credential = tokens.load(base, agent)
            payload = {'workspace': self.config['workspace'], 'name': name,
                       'model': model, 'harness': harness, 'project': project or {}}
            _status, body, _headers = self.peer._request(
                'POST', '/workspace/v1/ensure', data=json.dumps(payload).encode(),
                headers={'Content-Type': 'application/json',
                         'X-ML-Stack-Workspace-Token': credential})
            result = json.loads(body)
            if result.get('workspace') != self.config['workspace']:
                raise Denied('the responding coordinator workspace identity changed')
            tokens.store(base, result['agent'], result['token'])
            graph.upsert_node({'id': key, 'kind': 'remote-session', 'label': name,
                               'attrs': {'agent': result['agent']}})
            return result['agent']


def _device_peer(config):
    directory = home.state('onboard')
    active = {device.fingerprint for device in Devices(directory / 'devices.json').all()
              if device.status == 'active'}
    rows = [row for row in PeerBook(directory / 'peers.json').rows()
            if row.get('source') == 'pairing' and row.get('fingerprint') in active
            and row.get('certificate') == config.get('cert') and row.get('device_secret')]
    if len(rows) != 1:
        raise Denied('the coordinator has no unique active paired device credential')
    row = rows[0]
    secret = macauth.derive(base64.urlsafe_b64decode(row['device_secret']))
    http.pin(urlsplit(config['endpoint']).netloc, tls.pinned_context(row['certificate']))
    return Peer(config['endpoint'], secret, timeout=30)


def client(base):
    config = coordinator_config.load(base)
    if not config and load_cluster_key() is not None:
        candidates = [(peer, info) for peer, info in discover()]
        if len(candidates) > 1:
            raise Denied('several trusted coordinators are advertised; workspace routing is ambiguous')
        if not candidates:
            raise Denied('the enrolled workspace coordinator is unavailable; shared authority is preserved')
        peer, info = candidates[0]
        selected = {'mode': 'remote', 'workspace': info['workspace'], 'name': peer.name,
                    'endpoint': peer.base_url, 'cert': peer.beacon.cert if peer.beacon else ''}
        coordinator_config.validate_endpoint(selected['endpoint'])
        _device_peer(selected)
        config = coordinator_config.save(base, selected)
    if config.get('mode') != 'remote':
        return None
    return Remote(config, _device_peer(config))


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
    selected = {'mode': 'remote', 'workspace': info['workspace'],
                'name': peer.name, 'endpoint': peer.base_url,
                'cert': peer.beacon.cert if peer.beacon else ''}
    _device_peer(selected)
    return coordinator_config.save(base, selected)


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
