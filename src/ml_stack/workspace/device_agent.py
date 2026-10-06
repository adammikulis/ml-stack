"""Trusted installed-device accounts bind local worker identities independently of models."""
from ml_stack.graph.store import GraphStore
from ml_stack.home import device_id
from ml_stack.workspace import localagent, project, tokens
from ml_stack.workspace.chain import held
from ml_stack.workspace.device_accounts import account_for, account_in
from ml_stack.workspace.identity import AGENT, HUMAN, Denied
from ml_stack.workspace.onboard import TOKEN_S

__all__ = ["account_for", "bind_owned_worker", "bind_worker", "enroll"]


def _graph(ws):
    return GraphStore(ws.base / "device-accounts.db")


def enroll(ws, token):
    """Enroll this installed device once using actual person authority."""
    owner = ws.auth(token)
    if owner.role != HUMAN:
        raise Denied("only the person may enroll a device account")
    device = device_id()
    with held(ws.base / "device-accounts.lock"), _graph(ws) as graph:
        key = f"device:{device}"
        node = next((node for node in graph.nodes("device-account") if node["id"] == key), None)
        account = node["attrs"] if node else {}
        base = account.get("base_id", f"local-device-{device[:32]}")
        if not account and ws.registry.role_of(base):
            raise Denied("the device account identity is already owned by another enrollment")
        if not ws.registry.role_of(base):
            made = ws.registry.mint(owner, base, AGENT, TOKEN_S)
            tokens.store(ws.base, base, made)
        account = {"base_id": base, "device_id": device,
                   "enrolled_by": owner.id}
        graph.upsert_node({"id": key, "kind": "device-account", "label": base, "attrs": account})
        graph.upsert_node({"id": f"agent:{base}", "kind": "agent", "label": base})
        graph.upsert_node({"id": f"person:{owner.id}", "kind": "person", "label": owner.id})
        graph.upsert_edge({"source": f"person:{owner.id}", "target": key, "rel": "enrolled"})
        graph.upsert_edge({"source": f"agent:{base}", "target": key, "rel": "device-member"})
        account = account_in(graph, base)
    ws.audit("device-agent.enroll", owner.id, base_id=base, device_id=device)
    return account


def bind_worker(ws, token, name):
    """Bind an existing authenticated local worker to this person's device account."""
    owner = ws.auth(token)
    if owner.role != HUMAN:
        raise Denied("only the person may bind a device worker")
    worker = localagent.load(ws, name)
    if worker is None:
        raise ValueError("the local worker does not exist")
    identity = worker.identity or worker.name
    ws.auth(tokens.load(ws.base, identity))
    account = enroll(ws, token)
    with held(ws.base / "device-accounts.lock"), _graph(ws) as graph:
        old = account_in(graph, identity)
        if old and old["base_id"] != account["base_id"]:
            raise Denied("the worker already belongs to another enrolled device")
        graph.upsert_node({"id": f"agent:{identity}", "kind": "agent", "label": identity})
        graph.upsert_edge({"source": f"agent:{identity}", "target": f"device:{account['device_id']}",
                           "rel": "device-member"})
        account = account_in(graph, identity)
    ws.audit("device-agent.bind", owner.id, base_id=account["base_id"], agent=identity)
    return account


def _owned_parent(ws, token, device):
    parent = ws.auth(token)
    entry = ws.registry._load().get(parent.id, {})
    if parent.role != AGENT or parent.parent or entry.get('session_device'):
        raise Denied('local device membership requires an authenticated local launcher agent')
    for path in (ws.base, ws.registry.path, tokens.directory(ws.base),
                 tokens.directory(ws.base) / parent.id):
        if tokens.problem(path):
            raise Denied('local device membership requires private owned workspace state')
    if tokens.load(ws.base, parent.id) != token:
        raise Denied('local device membership requires the saved launcher session')
    existing = account_for(ws, parent.id)
    if existing and existing['device_id'] != device:
        raise Denied('the launcher belongs to another installed device')
    if entry.get('minted_by') != 'local-account' and not (
            existing and existing['device_id'] == device):
        raise Denied('the launcher has no trusted local device registration')
    return parent, entry


def bind_owned_worker(ws, token, name):
    """Bind an authenticated local launcher's delegated worker to the installed device."""
    device = device_id()
    parent, entry = _owned_parent(ws, token, device)
    worker = localagent.load(ws, name)
    if worker is None:
        raise ValueError('the local worker does not exist')
    identity = worker.identity or worker.name
    child = ws.auth(tokens.load(ws.base, identity))
    if child.role != AGENT or child.parent != parent.id:
        raise Denied('a local launcher binds only its own delegated worker')
    scope = project.describe(worker.project) if worker.project else {}
    if scope != entry.get('project', {}):
        raise Denied('the worker project differs from its launcher authorization')
    with held(ws.base / 'device-accounts.lock'), _graph(ws) as graph:
        key = f'device:{device}'
        node = next((node for node in graph.nodes('device-account') if node['id'] == key), None)
        account = node['attrs'] if node else {}
        base = account.get('base_id', f'local-device-{device[:32]}')
        registered = base in ws.registry.ids()
        if registered and (ws.registry.info(base)['role'] != AGENT or ws.registry.info(base)['revoked']):
            raise Denied('the installed device account was revoked')
        old = account_in(graph, identity)
        if old and old['base_id'] != base:
            raise Denied('the worker already belongs to another enrolled device')
        if not account:
            if registered:
                raise Denied('the device account identity belongs to another registration')
            account = {'base_id': base, 'device_id': device, 'enrolled_by': parent.id,
                       'source': 'owned-local-launcher'}
            graph.upsert_node({'id': key, 'kind': 'device-account', 'label': base, 'attrs': account})
        if not registered:
            if account.get('source') != 'owned-local-launcher':
                raise Denied('the installed device account registration is unavailable')
            with held(ws.registry.path.with_name('agents.lock')):
                if base in ws.registry.ids():
                    raise Denied('the device account identity belongs to another registration')
                credential = ws.registry._add('local-account', base, AGENT, 0)
            tokens.store(ws.base, base, credential)
        for member in (base, parent.id, identity):
            graph.upsert_node({'id': f'agent:{member}', 'kind': 'agent', 'label': member})
            graph.upsert_edge({'source': f'agent:{member}', 'target': key, 'rel': 'device-member'})
        account = account_in(graph, identity)
    ws.audit('device-agent.bind', parent.id, base_id=base, agent=identity, device_id=device)
    return account
