"""Task grants are assigned by genuine parents and revalidated against the live broker."""
import pytest
from workspace_kit import Kit

from ml_stack.graph.store import GraphStore
from ml_stack.workspace import device_agent, localagent, resource_allocations as resources, tokens
from ml_stack.workspace.identity import Denied


def test_shared_managed_holder_allowed_but_changed_grant_denied(tmp_path, monkeypatch):
    kit = Kit(tmp_path / "ws")
    monkeypatch.setattr(device_agent, "device_id", lambda: "1234567890abcdef")
    monkeypatch.setattr(resources, "device_id", lambda: "1234567890abcdef")
    parent = kit.agent("parent")
    child = kit.ws.delegate(parent, "worker")
    worker = child["id"]
    localagent.save(kit.ws, localagent.Agent("local-worker", "model", identity=worker, pid=555, process_started=42))
    device_agent.bind_worker(kit.ws, kit.owner, "local-worker")
    with GraphStore(kit.base / "coordination.db") as graph:
        graph.upsert_node({"id": "task:1234", "kind": "task", "label": "test"})
    status = {"servers": [{"ours": True, "pid": 999, "model": "model", "port": 51548,
        "holders": [{"lease": "lease", "pid": 555, "pid_started": 42}]}]}
    monkeypatch.setattr(resources.broker_wire, "status", lambda **kw: status)
    monkeypatch.setattr(resources, "pid_exists", lambda pid: True)
    monkeypatch.setattr(resources, "started_at", lambda pid: 42)
    status["servers"][0]["holders"][0]["pid"] = 666
    with pytest.raises(Denied, match="does not own"):
        resources.assign(kit.ws, parent, worker, "task:1234", "lease")
    status["servers"][0]["holders"][0]["pid"] = 555
    allocation = resources.assign(kit.ws, parent, worker, "task:1234", "lease")
    assert allocation["holder_pid"] == 555
    assert resources.verified_binding(kit.ws, worker, "task:1234", allocation["allocation_id"]) == allocation
    with pytest.raises(Denied, match="another worker"):
        resources.verified_binding(kit.ws, "foreign", "task:1234", allocation["allocation_id"])
    with pytest.raises(Denied, match="registered parent"):
        resources.assign(kit.ws, tokens.load(kit.base, worker), worker, "task:1234", "lease")
    status["servers"][0]["port"] = 51549
    with pytest.raises(Denied, match="grant changed"):
        resources.verified_binding(kit.ws, worker, "task:1234", allocation["allocation_id"])
    status["servers"][0]["unmanaged"] = True
    with pytest.raises(Denied, match="active managed"):
        resources.verified_binding(kit.ws, worker, "task:1234", allocation["allocation_id"])
