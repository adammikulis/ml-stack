"""Project agent enrollment bounds and scope checks."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from test_project_source import repository  # noqa: F401

from ml_stack.workspace.identity import AGENT, Denied, Registry
from ml_stack.workspace.remote_host import WorkspaceHost

PROJECT = "a" * 32
SCOPE = {"key": PROJECT, "name": "fixture-project", "cluster": "dev", "cluster_id": "c" * 64}


@pytest.fixture
def enrollment(tmp_path):
    project = SimpleNamespace(name="fixture-project", authority_machine="local", board_host="https://127.0.0.1:8770")
    projects = SimpleNamespace(machine="local", get=lambda identifier: project,
                               workspace_base=lambda identifier: tmp_path / identifier)
    host = WorkspaceHost(projects)
    body = {"name": "worker", "model": "gpt-6", "harness": "codex",
            "cluster": "dev", "cluster_id": "c" * 64, "authority_machine": "local"}
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
    code, reply = host.answer(PROJECT, "board", payload, cluster="dev", cluster_id="c" * 64, dev_admission=True)
    assert code == 200 and any(row["project"] == PROJECT for row in reply["result"])
    assert host.answer(PROJECT, "board", payload, cluster="prod")[0] == 403
    assert host.answer(PROJECT, "board", payload, cluster="dev", cluster_id="d" * 64)[0] == 403
    assert host.answer(PROJECT, "board", payload, cluster="dev", cluster_id="c" * 64)[0] == 403
    assert host.answer(PROJECT, "board", payload)[0] == 403
    assert not (ws.base / "tokens" / ".owner").exists()


@pytest.mark.parametrize("field,value", [("cluster", "prod"), ("authority_machine", "foreign"),
                                          ("model", "bad model"), ("harness", "bad harness"),
                                          ("name", "owner")])
def test_enrollment_rejects_invalid_claims(enrollment, field, value):
    host, _, body = enrollment
    code, _ = host.enroll(PROJECT, {**body, field: value}, cluster="dev", cluster_id="c" * 64)
    assert code in {400, 403}
    assert not host.workspace(PROJECT).registry.ids()


def test_foreign_authority_is_not_reassigned(enrollment):
    host, project, body = enrollment
    project.authority_machine = "foreign"
    assert host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)[0] == 403
    assert project.authority_machine == "foreign"


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
    code, reply = host.answer(PROJECT, "board", request, cluster="dev", cluster_id="c" * 64, dev_admission=True)
    assert code == 200, reply
    child = reply["result"]
    ws = host.workspace(PROJECT)
    identity = ws.auth(child["token"])
    assert identity.id == parent["id"] + "/native" and identity.role == AGENT
    assert identity.can == ws.auth(parent["token"]).can
    assert ws.registry.info(identity.id)["expires"] <= ws.registry.info(parent["id"])["expires"]
    request = {"agent_token": child["token"], "operation": "board.list"}
    code, reply = host.answer(PROJECT, "board", request, cluster="dev", cluster_id="c" * 64, dev_admission=True)
    assert code == 200 and any(row["project"] == PROJECT for row in reply["result"])
    request["operation"], request["args"] = "delegate", ["another"]
    assert host.answer(PROJECT, "board", request, cluster="dev", cluster_id="c" * 64, dev_admission=True)[0] == 403
    request["operation"], request["args"] = "revoke_self", [parent["id"]]
    assert host.answer(PROJECT, "board", request, cluster="dev", cluster_id="c" * 64, dev_admission=True)[0] == 400
    request["args"] = []
    assert host.answer(PROJECT, "board", request, cluster="dev", cluster_id="c" * 64, dev_admission=True)[0] == 200
    with pytest.raises(Denied, match="revoked"):
        ws.auth(child["token"])
    assert ws.auth(parent["token"]).id == parent["id"]


def test_native_parent_delegation_is_bounded(enrollment):
    host, _, body = enrollment
    _, parent = host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)
    for index in range(8):
        request = {"agent_token": parent["token"], "operation": "delegate", "args": [f"native-{index}"]}
        assert host.answer(PROJECT, "board", request, cluster="dev", cluster_id="c" * 64, dev_admission=True)[0] == 200
    request["args"] = ["overflow"]
    assert host.answer(PROJECT, "board", request, cluster="dev", cluster_id="c" * 64, dev_admission=True)[0] == 403


def test_native_reservations_are_atomic_project_scoped_and_return_relative_keys(enrollment, repository):  # noqa: F811
    host, project, body = enrollment
    project.root = str(repository)
    _, first = host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)
    _, second = host.enroll(PROJECT, {**body, "name": "other"}, cluster="dev", cluster_id="c" * 64)
    def request(agent, operation, *args, **kwargs):
        return host.answer(PROJECT, "board", {"agent_token": agent["token"], "operation": operation,
                           "args": list(args), "kwargs": kwargs}, cluster="dev", cluster_id="c" * 64, dev_admission=True)
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
