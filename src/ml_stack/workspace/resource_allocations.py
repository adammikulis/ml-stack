"""Authenticated scheduler allocations reference live maintained broker grants."""
import os
import sys
import uuid

from ml_stack.graph.store import GraphStore
from ml_stack.home import device_id
from ml_stack.serve import broker_wire
from ml_stack.serve.process import pid_exists, started_at
from ml_stack.workspace import localagent, task_authority, task_worktrees, tokens
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
    workers = [localagent.load(ws, name) for name in localagent.names(ws)]
    found = [worker for worker in workers if worker and (worker.identity or worker.name) == identity]
    if len(found) != 1:
        raise Denied("the allocation requires one registered local worker")
    return found[0]


def _execution_config(runner):
    return {key: getattr(runner, key) for key in
            ('model', 'harness', 'ctx', 'effort', 'max_effort', 'role', 'profile', 'project')}


def assign(ws, token, worker, task, lease_id):
    """The actual person or registered parent assigns a live grant to an enrolled worker."""
    caller = ws.auth(token)
    child = ws.auth(tokens.load(ws.base, worker))
    if caller.role != HUMAN and child.parent != caller.id:
        task_authority.authorize(ws, token, worker, task)
    account = account_for(ws, worker)
    if account is None or account["device_id"] != device_id():
        raise Denied("the worker has no person-enrolled account on this device")
    grant = _grant(lease_id, broker_wire.status(start=False))
    runner = _worker(ws, worker)
    if caller.role != HUMAN and (runner.pid != grant["holder_pid"]
                                 or runner.process_started != grant["holder_started"]):
        raise Denied("the worker does not own this broker holder")
    if runner.profile not in ("chat", "coding"):
        raise Denied("the configured worker profile has no coordinator capability")
    allocation = {"allocation_id": f"allocation:{uuid.uuid4().hex}", "worker": worker, "task": task,
                  "device_id": account["device_id"], "base_id": account["base_id"], **grant,
                  "owner": caller.id, "person_assigned": caller.role == HUMAN, "requester": worker, "reason": f"Task {task}",
                  "profile": runner.profile, "capabilities": [runner.profile], "harness": runner.harness,
                  "execution_config": _execution_config(runner),
                  "project": runner.project, "interpreter": sys.executable, "python": sys.version.split()[0], "scheduler_pid": os.getpid()}
    if runner.profile == 'coding':
        task_worktrees.prepare(ws, token, worker, task)
    with held(ws.base / "coordination.lock"), GraphStore(ws.base / "coordination.db") as graph:
        if not graph.has(task):
            raise ValueError("the canonical task does not exist")
        if runner.profile == 'coding':
            scope = task_worktrees.binding(graph, worker, task)
            allocation.update({key: scope[key] for key in ('project', 'source_project', 'baseline_commit')})
        existing = next((row['attrs'] for row in graph.nodes('allocation')
                         if row['attrs']['task'] == task and row['attrs']['worker'] == worker
                         and row['attrs']['lease_id'] == lease_id), None)
        if existing and all(existing.get(key) == allocation.get(key) for key in
                            ('owner', 'holder_pid', 'holder_started', 'server_pid', 'server_started', 'project', 'baseline_commit', 'execution_config')):
            return existing
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
    if allocation.get('execution_config') != _execution_config(runner):
        raise Denied('the worker execution configuration changed after allocation')
    if runner.profile != allocation["profile"] or runner.project != allocation.get("source_project", allocation["project"]):
        raise Denied("the worker configuration changed after allocation")
    if not allocation["person_assigned"] and (runner.pid != grant["holder_pid"]
                                              or runner.process_started != grant["holder_started"]):
        raise Denied("the worker no longer owns this broker holder")
    return allocation
