"""Person-enrolled device accounts bind local worker identities independently of models."""
from ml_stack.graph.store import GraphStore
from ml_stack.home import device_id
from ml_stack.workspace import localagent, tokens
from ml_stack.workspace.chain import held
from ml_stack.workspace.device_accounts import account_for, account_in
from ml_stack.workspace.identity import AGENT, HUMAN, Denied
from ml_stack.workspace.onboard import TOKEN_S

__all__ = ["account_for", "bind_worker", "enroll"]


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

