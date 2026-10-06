"""Standard agent sessions bound to authenticated paired devices and hosted projects."""

import hashlib
import secrets

from ml_stack.graph.store import GraphStore
from ml_stack.workspace import onboard, tokens
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import AGENT, CAPS, PREFIX, Denied, valid_name
from ml_stack.workspace.modelid import CLAIMED, clean_harness, clean_model


def _project(projects, requested):
    if projects is None or type(requested) is not dict or set(requested) - {'key', 'name'}:
        raise Denied('agent registration requires an existing hosted project')
    candidates = [row for row in projects.list()
                  if row['authority_machine'] == projects.machine
                  and (row['id'] == requested['key'] if requested.get('key')
                       else row['name'] == requested.get('name'))]
    if len(candidates) != 1:
        raise Denied('the requested project is not uniquely hosted by this coordinator')
    row = candidates[0]
    return {'key': row['id'], 'name': row['name']}


def _credential(ws, graph, binding, token):
    agent, scope, fingerprint = (binding[key] for key in ('agent', 'project', 'device'))
    with held(ws.registry.path.with_name('agents.lock')):
        agents = ws.registry._load()
        entry = agents.get(agent)
        if entry and (entry.get('revoked') or entry.get('role') != AGENT
                      or entry.get('session_device') != fingerprint
                      or entry.get('project') != scope):
            raise Denied('the registered agent identity is unavailable')
        credential = ''
        if token:
            try:
                actor = ws.auth(token)
            except Denied:
                actor = None
            if actor and actor.id != agent:
                raise Denied('the presented session belongs to another agent')
            if actor:
                credential = token
        if not credential:
            key = 'session:' + hashlib.sha256(f"{fingerprint}:{binding['requested']}".encode()).hexdigest()
            graph.upsert_node({'id': key, 'kind': 'device-session', 'label': agent, 'attrs': binding})
            value = secrets.token_urlsafe(32)
            now = ws.clock()
            if entry is None:
                entry = {'role': AGENT, 'created': now, 'minted_by': 'paired-device',
                         'can': list(CAPS), 'project': scope, 'session_device': fingerprint,
                         'revoked': False}
                agents[agent] = entry
            entry.update(hash=hashlib.sha256(value.encode()).hexdigest(), expires=now + onboard.TOKEN_S)
            ws.registry._save(agents)
            credential = f'{PREFIX}{agent}.{value}'
    return credential


def ensure(ws, device, projects, document, token=''):
    if device is None or device.status != 'active':
        raise Denied('agent registration requires an active paired device')
    name, model, harness = (document.get(key, '') for key in ('name', 'model', 'harness'))
    if any(type(value) is not str for value in (name, model, harness)):
        raise ValueError('agent identity and model labels are strings')
    if not valid_name(name) or name in onboard.JOIN_RESERVED:
        raise Denied('choose a standard agent identity')
    harness = clean_harness(harness)
    if model:
        clean_model(model)
    fingerprint = device.fingerprint
    requested_scope = _project(projects, document.get('project'))
    with held(ws.base / 'device-sessions.lock'), GraphStore(ws.base / 'device-sessions.db') as graph:
        bindings = [node['attrs'] for node in graph.nodes('device-session')]
        binding = next((row for row in bindings if row['device'] == fingerprint
                        and name in (row['requested'], row['agent'])), None)
        if binding:
            agent, scope = binding['agent'], binding['project']
            live_scope = _project(projects, scope)
            if live_scope != scope:
                raise Denied('the agent project authorization changed')
            if requested_scope != scope:
                raise Denied('the requested project differs from the registered agent scope')
        else:
            if not device.mine:
                raise Denied('this paired device has no existing agent authorization')
            if (len(bindings) >= ws.limits.agents_live
                    or sum(row['device'] == fingerprint for row in bindings) >= ws.limits.mints_per_identity):
                raise Denied('the registered device agent limit is reached')
            scope = requested_scope
            agent = name
            if agent in ws.registry.ids():
                agent = name[:35] + '-' + fingerprint[:12]
            if agent in ws.registry.ids():
                raise Denied('this agent identity belongs to another device registration')
            binding = {'device': fingerprint, 'requested': name, 'agent': agent, 'project': scope}
        credential = _credential(ws, graph, binding, token)
        tokens.store(ws.base, agent, credential)
        ws.board.place(agent, scope)
        if model or harness:
            ws.registry.record_model(agent, model, harness, CLAIMED)
        ws.audit('device-session.ensure', agent, device=fingerprint, project=scope['key'])
        return agent, credential


def check(ws, token, device, projects):
    actor = ws.auth(token)
    agents = ws.registry._load()
    root = agents.get(ws.registry._root(agents, actor.id), {})
    entry = agents.get(actor.id, {})
    fingerprint = root.get('session_device') or entry.get('session_device')
    if fingerprint:
        if device is None or device.status != 'active' or fingerprint != device.fingerprint:
            raise Denied('the agent session requires its active paired device')
        scope = root.get('project', {})
        if _project(projects, scope) != scope:
            raise Denied('the agent project authorization changed')
        if entry.get('project') and entry['project'] != scope:
            raise Denied('the delegated agent project differs from its root authorization')
