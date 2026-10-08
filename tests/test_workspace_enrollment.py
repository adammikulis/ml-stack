"""Project agent enrollment bounds and scope checks."""
import io
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from test_project_source import repository  # noqa: F401

from ml_stack.workspace import remote_task_client
from ml_stack.workspace.identity import AGENT, Denied, Registry
from ml_stack.workspace.remote_host import WorkspaceHost
from ml_stack.workspace.taskboard import TaskBoard

PROJECT = "a" * 32
SCOPE = {"key": PROJECT, "name": "fixture-project", "cluster": "dev", "cluster_id": "c" * 64}


@pytest.fixture
def enrollment(tmp_path):
    project = SimpleNamespace(id=PROJECT, root=str(tmp_path), name="fixture-project", board_host="https://127.0.0.1:8770")
    projects = SimpleNamespace(machine="local", hosts=lambda host: host == "https://127.0.0.1:8770", get=lambda identifier: project,
                               workspace_base=lambda identifier: tmp_path / identifier)
    host = WorkspaceHost(projects)
    body = {"name": "worker", "model": "gpt-6", "harness": "codex",
            "cluster": "dev", "cluster_id": "c" * 64}
    return host, project, body


def test_enrollment_bootstraps_only_project_agents_and_places_board(enrollment):
    host, _, body = enrollment
    code, first = host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)
    assert code == 201, first
    code, second = host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)
    assert code == 201 and second["id"] != first["id"]
    ws = host.workspace(PROJECT)
    assert all(ws.registry.info(name)["role"] == AGENT for name in ws.registry.ids())
    assert ws.registry.info(first["id"])["project"] == SCOPE
    assert ws.registry.info(first["id"])["model_state"] == "claimed"
    payload = {"agent_token": first["token"], "operation": "board.list"}
    code, reply = host.answer(PROJECT, "board", payload, admission=("dev", "c" * 64, True))
    assert code == 200
    board = next(row for row in reply["result"] if row["name"] == "#fixture-project")
    assert board["project"] is True and board["member"] is True
    assert ws.board.store.state()[0][board["name"]]["project"] == PROJECT
    assert host.answer(PROJECT, "board", payload, admission=("prod", "c" * 64, True))[0] == 403
    assert host.answer(PROJECT, "board", payload, admission=("dev", "d" * 64, True))[0] == 403
    assert host.answer(PROJECT, "board", payload, admission=("dev", "c" * 64, False))[0] == 403
    assert host.answer(PROJECT, "board", payload)[0] == 403
    assert not (ws.base / "tokens" / ".owner").exists()


@pytest.mark.parametrize("field,value", [("cluster", "prod"),
                                          ("model", "bad model"), ("harness", "bad harness"),
                                          ("name", "owner")])
def test_enrollment_rejects_invalid_claims(enrollment, field, value):
    host, _, body = enrollment
    code, _ = host.enroll(PROJECT, {**body, field: value}, cluster="dev", cluster_id="c" * 64)
    assert code in {400, 403}
    assert not host.workspace(PROJECT).registry.ids()


def test_foreign_authority_is_not_reassigned(enrollment):
    host, project, body = enrollment
    project.board_host = "https://foreign:8770"
    assert host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)[0] == 403
    assert project.board_host == "https://foreign:8770"


def test_enrollment_rate_is_shared_across_claimed_names(enrollment):
    host, _, body = enrollment
    for index in range(10):
        assert host.enroll(PROJECT, {**body, "name": f"worker-{index}"}, cluster="dev", cluster_id="c" * 64)[0] == 201
    assert host.enroll(PROJECT, {**body, "name": "other-worker"}, cluster="dev", cluster_id="c" * 64)[0] == 429


def test_atomic_registry_enrollment_cannot_reuse_names_or_exceed_live_bound(tmp_path):
    registry = Registry(tmp_path, clock=lambda: 100)
    with ThreadPoolExecutor(max_workers=8) as executor:
        issued = list(executor.map(lambda _: registry.enroll_project("worker", SCOPE, 300), range(64)))
    assert len({name for name, _ in issued}) == 64
    assert all(registry.authenticate(token).role == AGENT for _, token in issued)
    with pytest.raises(Denied, match="64 live"):
        registry.enroll_project("worker", SCOPE, 300)
    expired = Registry(tmp_path, clock=lambda: 401)
    name, _ = expired.enroll_project("worker", SCOPE, 300)
    assert name != "worker"


@pytest.mark.parametrize("ttl", [0, -1, float("nan"), float("inf"), 30 * 86400 + 1])
def test_registry_enrollment_refuses_unbounded_lifetimes(tmp_path, ttl):
    registry = Registry(tmp_path)
    with pytest.raises(ValueError, match="lifetime"):
        registry.enroll_project("worker", SCOPE, ttl)
    assert registry.ids() == []


def test_native_child_delegation_inherits_scope_and_self_revocation(enrollment):
    host, _, body = enrollment
    _, parent = host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)
    request = {"agent_token": parent["token"], "operation": "delegate", "args": ["native"]}
    code, reply = host.answer(PROJECT, "board", request, admission=("dev", "c" * 64, True))
    assert code == 200, reply
    child = reply["result"]
    ws = host.workspace(PROJECT)
    identity = ws.auth(child["token"])
    assert identity.id == parent["id"] + "/native" and identity.role == AGENT
    assert identity.can == ws.auth(parent["token"]).can
    assert ws.registry.info(identity.id)["expires"] <= ws.registry.info(parent["id"])["expires"]
    request = {"agent_token": child["token"], "operation": "board.list"}
    code, reply = host.answer(PROJECT, "board", request, admission=("dev", "c" * 64, True))
    assert code == 200
    board = next(row for row in reply["result"] if row["name"] == "#fixture-project")
    assert board["project"] is True and board["member"] is True
    assert ws.board.store.state()[0][board["name"]]["project"] == PROJECT
    request["operation"], request["args"] = "delegate", ["another"]
    assert host.answer(PROJECT, "board", request, admission=("dev", "c" * 64, True))[0] == 403
    request["operation"], request["args"] = "revoke_self", [parent["id"]]
    assert host.answer(PROJECT, "board", request, admission=("dev", "c" * 64, True))[0] == 400
    request["args"] = []
    assert host.answer(PROJECT, "board", request, admission=("dev", "c" * 64, True))[0] == 200
    with pytest.raises(Denied, match="revoked"):
        ws.auth(child["token"])
    assert ws.auth(parent["token"]).id == parent["id"]


def test_native_parent_delegation_is_bounded(enrollment):
    host, _, body = enrollment
    _, parent = host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)
    for index in range(8):
        request = {"agent_token": parent["token"], "operation": "delegate", "args": [f"native-{index}"]}
        assert host.answer(PROJECT, "board", request, admission=("dev", "c" * 64, True))[0] == 200
    request["args"] = ["overflow"]
    assert host.answer(PROJECT, "board", request, admission=("dev", "c" * 64, True))[0] == 403


def test_native_reservations_are_atomic_project_scoped_and_return_relative_keys(enrollment, repository):  # noqa: F811
    host, project, body = enrollment
    project.root = str(repository)
    _, first = host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)
    _, second = host.enroll(PROJECT, {**body, "name": "other"}, cluster="dev", cluster_id="c" * 64)
    def request(agent, operation, *args, **kwargs):
        return host.answer(PROJECT, "board", {"agent_token": agent["token"], "operation": operation,
                           "args": list(args), "kwargs": kwargs}, admission=("dev", "c" * 64, True))
    code, reply = request(first, "native.reserve", [["area", "src/item.py"], ["branch", "feature"]], label="native")
    assert code == 200, reply
    assert {row["key"] for row in reply["result"]} == {"src/item.py", "feature"}
    assert all(row["pid"] == 0 and row["owner"] == first["id"] for row in reply["result"])
    code, listed = request(first, "claims")
    assert code == 200 and "src/item.py" in {row["key"] for row in listed["result"]}
    assert str(repository) not in str(listed)
    assert request(first, "who", "area", "src/item.py")[1]["result"]["key"] == "src/item.py"
    code, reply = request(second, "native.reserve", [["branch", "other"], ["area", "src"]])
    assert code == 409 and str(repository) not in str(reply)
    assert not any(row["key"] == "other" for row in host.workspace(PROJECT).claims.listing())
    code, reply = request(second, "native.release", "area", "src/item.py")
    assert code == 403 and str(repository) not in str(reply)
    assert request(first, "native.release", "area", "src/item.py")[0] == 200
    for path in ("../outside", "/absolute", "src/../item", "src\\item", "src//item"):
        assert request(first, "native.reserve", [["area", path]])[0] == 400
    assert request(first, "native.reserve", [["port", "8000"]])[0] == 400
    for branch in ("feature/", "feature.lock", "feature//child"):
        assert request(first, "native.reserve", [["branch", branch]])[0] == 400
    assert request(first, "native.reserve", [["branch", "valid"]], pid=9999)[0] == 400


def test_self_revocation_cleans_exact_claims_and_refuses_replaced_generation(enrollment):
    host, _, body = enrollment
    _, parent = host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)
    ws = host.workspace(PROJECT)
    actor = ws.auth(parent["token"])
    old = ws.registry.delegate(actor, "native", 300, actor.can, 8)
    child = ws.auth(old)
    ws.claims.reserve(child, [("branch", "child-work")])
    ws.claims.reserve(actor, [("branch", "parent-work")])
    ws.registry.revoke_self(old, lambda who: host._release_owned_claims(ws, who))
    assert {row["owner"] for row in ws.claims.listing()} == {actor.id}
    replacement = ws.registry.delegate(actor, "native", 300, actor.can, 8)
    with pytest.raises(Denied, match="recognised"):
        ws.registry.revoke_self(old, lambda who: host._release_owned_claims(ws, who))
    assert ws.auth(replacement).id == child.id


def test_typed_task_commands_bind_full_project_scope_and_idempotency(enrollment):
    host, _, body = enrollment
    _, agent = host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)
    def command(action, payload, request_id="e" * 32):
        return host.answer(PROJECT, "board", {"agent_token": agent["token"], "operation": "task.command",
                           "args": [{"action": action, "payload": payload, "request_id": request_id}]},
                           admission=("dev", "c" * 64, True))
    spec = {"title": "Typed task", "acceptance": ["Scoped outcome"]}
    code, created = command("task-create", {"spec": spec})
    assert code == 200, created
    ident = created["result"]["id"]
    assert created["result"]["project"] == SCOPE
    assert command("task-create", {"spec": spec})[1]["result"]["id"] == ident
    assert command("task-create", {"spec": {**spec, "title": "Different"}})[0] == 403
    assert command("task-create", {"spec": {**spec, "project": {"key": "b" * 32}}}, "f" * 32)[0] == 403
    ws = host.workspace(PROJECT)
    foreign = TaskBoard(ws).create(agent["token"], {**spec, "project": {"key": "b" * 32}})
    assert command("task", {"id": foreign["id"]})[0] == 403
    assert {row["id"] for row in command("tasks", {})[1]["result"]["tasks"]} == {ident}
    assert command("task-create", {"spec": {**spec, "deps": [foreign["id"]]}}, "f" * 32)[0] == 403
    assert command("task-claim", {"id": ident, "allocation_id": "untrusted"}, "f" * 32)[0] == 403
    for action in ("task-integrate", "resource-assign", "task-recover", "mint"):
        assert command(action, {})[0] == 400


def test_native_heartbeat_does_not_renew_physical_or_foreign_area_claims(enrollment, repository):  # noqa: F811
    host, project, body = enrollment
    project.root = str(repository / "project")
    _, agent = host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)
    ws = host.workspace(PROJECT)
    actor = ws.auth(agent["token"])
    ws.claims.reserve(actor, [("branch", "source"), ("area", str(repository / "outside")),
                              ("port", "9000"), ("worktree", str(repository / "physical"))])
    before = {(row["kind"], row["key"]): row["expires"] for row in ws.claims.listing()}
    code, reply = host.answer(PROJECT, "board", {"agent_token": agent["token"], "operation": "native.heartbeat",
                             "kwargs": {"ttl_s": 1800}}, admission=("dev", "c" * 64, True))
    assert code == 200, reply
    assert [row["kind"] for row in reply["result"]] == ["branch"]
    after = {(row["kind"], row["key"]): row["expires"] for row in ws.claims.listing()}
    assert after[("branch", "source")] > before[("branch", "source")]
    assert all(after[key] == expiry for key, expiry in before.items() if key[0] != "branch")


def test_canonical_task_client_preserves_stable_request_and_json_stdin(monkeypatch):
    calls = []
    remote = SimpleNamespace(call=lambda *args: calls.append(args) or {"id": "task:" + "a" * 32})
    args = SimpleNamespace(cmd="task-create", payload="-", request_id="e" * 32)
    for _ in range(2):
        monkeypatch.setattr("sys.stdin", io.StringIO('{"title":"Typed task","acceptance":["Scoped outcome"]}'))
        assert remote_task_client.command(remote, "private-agent-capability", args)["id"] == "task:" + "a" * 32
    assert calls[0] == calls[1]
    operation, token, document = calls[0]
    assert operation == "task.command" and token == "private-agent-capability"
    assert document["request_id"] == "e" * 32 and document["action"] == "task-create"
    assert document["payload"] == {"spec": {"title": "Typed task", "acceptance": ["Scoped outcome"]}}


def test_canonical_task_client_claim_maps_only_existing_allocation_and_refuses_integration():
    calls = []
    remote = SimpleNamespace(call=lambda *args: calls.append(args) or {})
    args = SimpleNamespace(cmd="task-claim", id="task:" + "a" * 32,
                           allocation_id="allocation:" + "b" * 32, request_id="e" * 32)
    remote_task_client.command(remote, "private-agent-capability", args)
    document = calls[0][2]
    assert document["action"] == "task-claim"
    assert document["payload"] == {"id": args.id, "allocation_id": args.allocation_id}
    args.cmd = "task-integrate"
    with pytest.raises(Denied, match="unavailable"):
        remote_task_client.command(remote, "private-agent-capability", args)
    assert len(calls) == 1


def test_new_agent_id_enrolls_without_a_prior_selection(enrollment):
    host, _, body = enrollment
    code, issued = host.enroll(PROJECT, {**body, "name": "claude", "harness": "claude-code"},
                               cluster="dev", cluster_id="c" * 64)
    assert code == 201, issued
    assert issued["id"].startswith("claude")
    assert host.workspace(PROJECT).registry.info(issued["id"])["role"] == AGENT


@pytest.mark.redteam
def test_board_hosted_elsewhere_refuses_enrollment_and_renewal(enrollment):
    host, project, body = enrollment
    code, issued = host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)
    assert code == 201, issued
    project.board_host = "https://foreign:8770"
    assert host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)[0] == 403
    renewal = {"agent_token": issued["token"], "cluster": "dev", "cluster_id": "c" * 64}
    assert host.renew(PROJECT, renewal, cluster="dev", cluster_id="c" * 64)[0] == 403
