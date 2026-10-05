"""Authenticated scheduler allocations reference live maintained broker grants."""
import os
import sys
import uuid

from ml_stack.graph.store import GraphStore
from ml_stack.home import device_id
from ml_stack.serve import broker_wire
from ml_stack.serve.process import pid_exists, started_at
from ml_stack.workspace import tokens
from ml_stack.workspace.chain import held
from ml_stack.workspace.device_accounts import account_for
from ml_stack.workspace.identity import HUMAN, Denied


def _grant(lease_id, status):
    for server in status.get("servers", []):
        if server.get("unmanaged") or not server.get("ours") or server.get("loading"):
            continue
        for holder in server.get("holders", []):
            if holder.get("lease") == lease_id:
                pid, born = holder.get("pid"), holder.get("pid_started")
                if not pid_exists(pid) or started_at(pid) != born or not pid_exists(server.get("pid")):
                    raise Denied("the broker allocation holder is no longer live")
                return {"lease_id": lease_id, "holder_pid": pid, "holder_started": born,
                        "server_pid": server["pid"], "server_started": started_at(server["pid"]), "port": server["port"], "model": server["model"]}
    raise Denied("the resource lease is not an active managed broker grant")


def _worker(ws, identity):
    from ml_stack.workspace import localagent
    workers = [localagent.load(ws, name) for name in localagent.names(ws)]
    found = [worker for worker in workers if worker and (worker.identity or worker.name) == identity]
    if len(found) != 1:
        raise Denied("the allocation requires one registered local worker")
    return found[0]


def assign(ws, token, worker, task, lease_id):
    """The actual person or registered parent assigns a live grant to an enrolled worker."""
    caller = ws.auth(token)
    child = ws.auth(tokens.load(ws.base, worker))
    if caller.role != HUMAN and child.parent != caller.id:
        raise Denied("only the person or the worker's registered parent assigns resources")
    account = account_for(ws, worker)
    if account is None or account["device_id"] != device_id():
        raise Denied("the worker has no person-enrolled account on this device")
    grant = _grant(lease_id, broker_wire.status(start=False))
    runner = _worker(ws, worker)
    if caller.role != HUMAN and (runner.pid != grant["holder_pid"]
                                 or runner.process_started != grant["holder_started"]):
        raise Denied("the worker does not own this broker holder")
    allocation = {"allocation_id": f"allocation:{uuid.uuid4().hex}", "worker": worker, "task": task,
                  "device_id": account["device_id"], "base_id": account["base_id"], **grant,
                  "owner": caller.id, "person_assigned": caller.role == HUMAN, "requester": worker, "reason": f"Task {task}",
                  "interpreter": sys.executable, "python": sys.version.split()[0], "scheduler_pid": os.getpid()}
    with held(ws.base / "coordination.lock"), GraphStore(ws.base / "coordination.db") as graph:
        if not graph.has(task):
            raise ValueError("the canonical task does not exist")
        key = allocation["allocation_id"]
        graph.upsert_node({"id": key, "kind": "allocation", "label": task, "attrs": allocation})
        for target, kind, rel in ((f"agent:{worker}", "agent", "allocated-to"),
                                  (f"device:{account['device_id']}", "device", "reserved-on"),
                                  (f"agent:{caller.id}", "agent", "assigned-by")):
            graph.upsert_node({"id": target, "kind": kind, "label": target})
            graph.upsert_edge({"source": key, "target": target, "rel": rel})
        graph.upsert_edge({"source": key, "target": task, "rel": "for-task"})
    ws.audit("resource.assign", caller.id, allocation_id=key, worker=worker, task=task)
    return allocation


def verified_binding(ws, worker, task, allocation_id, *, status=None):
    """Verify a persisted assignment against the broker's current live holder and enrollment."""
    with GraphStore(ws.base / "coordination.db") as graph:
        allocation = next((node["attrs"] for node in graph.nodes("allocation")
                           if node["id"] == allocation_id), None)
    if not allocation or allocation["worker"] != worker or allocation["task"] != task:
        raise Denied("the allocation belongs to another worker or task")
    account = account_for(ws, worker)
    if account is None or any(account[k] != allocation[k] for k in ("base_id", "device_id")):
        raise Denied("the allocation's device enrollment changed")
    grant = _grant(allocation["lease_id"], (status or broker_wire.status)(start=False))
    if any(allocation[k] != value for k, value in grant.items()):
        raise Denied("the broker grant changed after resource assignment")
    runner = _worker(ws, worker)
    if not allocation["person_assigned"] and (runner.pid != grant["holder_pid"]
                                              or runner.process_started != grant["holder_started"]):
        raise Denied("the worker no longer owns this broker holder")
    return allocation
