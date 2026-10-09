"""Automatic Dev capabilities follow authenticated cluster key changes."""

import hashlib
import json
import ssl
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from poolhouse.fleet import api, project_enrollment
from poolhouse.fleet.discovery import derive_token
from poolhouse.workspace import automatic_connection, project_connection, tokens
from poolhouse.workspace.identity import Denied
from poolhouse.workspace.remote import RemoteWorkspace

pytest_plugins = ("test_workspace_enrollment",)

PROJECT = "a" * 32
CURRENT = "d" * 64


def enrolled(enrollment):
    host, project, body = enrollment
    code, issued = host.enroll(PROJECT, body, cluster="dev", cluster_id="c" * 64)
    assert code == 201, issued
    request = {"agent_token": issued["token"], "cluster": "fresh-dev", "cluster_id": CURRENT}
    return host, project, issued, request


def test_renewal_preserves_capability_identity_history_and_membership(enrollment):
    host, _, issued, request = enrolled(enrollment)
    ws = host.workspace(PROJECT)
    before = ws.registry._load()[issued["id"]]
    history = ws.board.store.log.rows()
    query = {"agent_token": issued["token"], "operation": "whoami"}
    assert host.answer(PROJECT, "board", query, admission=("fresh-dev", CURRENT, True))[0] == 403
    assert host.renew(PROJECT, request, cluster="fresh-dev", cluster_id=CURRENT)[0] == 200
    after = ws.registry._load()[issued["id"]]
    assert {k: v for k, v in before.items() if k != "project"} == {
        k: v for k, v in after.items() if k != "project"}
    assert after["project"] == {**before["project"], "cluster": "fresh-dev", "cluster_id": CURRENT}
    assert ws.board.store.log.rows() == history
    code, reply = host.answer(PROJECT, "board", query, admission=("fresh-dev", CURRENT, True))
    assert code == 200 and reply["result"]["id"] == issued["id"]
    assert host.answer(PROJECT, "board", query, admission=("dev", "c" * 64, True))[0] == 403


@pytest.mark.redteam
@pytest.mark.parametrize("attack", ["invited", "person", "device", "child", "revoked", "expired", "project", "authority"])
def test_renewal_never_restores_other_authority(enrollment, attack):
    host, project, issued, request = enrolled(enrollment)
    ws = host.workspace(PROJECT)
    rows = ws.registry._load()
    entry = rows[issued["id"]]
    if attack == "invited":
        entry["minted_by"] = "person-issued"
    elif attack == "person":
        entry["role"] = "human"
    elif attack == "device":
        entry["session_device"] = "e" * 64
    elif attack == "child":
        request["agent_token"] = ws.registry.delegate(ws.auth(issued["token"]), "child", 300, (), 8)
        rows = ws.registry._load()
    elif attack == "revoked":
        entry["revoked"] = True
    elif attack == "expired":
        entry["expires"] = 1
    elif attack == "project":
        entry["project"]["key"] = "b" * 32
    else:
        project.board_host = "https://foreign:8770"
    ws.registry._save(rows)
    before = ws.registry.path.read_bytes()
    assert host.renew(PROJECT, request, cluster="fresh-dev", cluster_id=CURRENT)[0] == 403
    assert ws.registry.path.read_bytes() == before


def test_registry_authenticates_renewal_inside_revocation_lock(enrollment, monkeypatch):
    host, _, issued, request = enrolled(enrollment)
    registry = host.workspace(PROJECT).registry
    authenticated, release, revoking, revoked = (threading.Event() for _ in range(4))
    authenticate = registry.authenticate
    actor = authenticate(issued["token"])
    def paused_auth(token):
        who = authenticate(token)
        authenticated.set()
        assert release.wait(5)
        return who
    monkeypatch.setattr(registry, "authenticate", paused_auth)
    results = []
    def revoke():
        revoking.set()
        registry.revoke(actor, actor.id)
        revoked.set()
    renewal = threading.Thread(target=lambda: results.append(registry.renew_dev_project(
        issued["token"], PROJECT, request["cluster"], CURRENT)))
    revocation = threading.Thread(target=revoke)
    renewal.start()
    try:
        assert authenticated.wait(5)
        revocation.start()
        assert revoking.wait(5)
        assert not revoked.wait(0.1)
    finally:
        release.set()
        renewal.join(5)
        if revocation.ident:
            revocation.join(5)
    assert len(results) == 1 and revoked.is_set()
    with pytest.raises(Denied, match="revoked"):
        registry.renew_dev_project(issued["token"], PROJECT, request["cluster"], CURRENT)


@pytest.mark.redteam
@pytest.mark.parametrize("attack", ["plain", "unsealed", "prod", "wrong-key"])
def test_renew_route_requires_current_active_dev_tls(tmp_path, monkeypatch, attack):
    member = SimpleNamespace(group="fresh-dev", key=b"current-key", mode="prod" if attack == "prod" else "dev")
    monkeypatch.setattr(project_enrollment, "memberships", lambda path: [member])
    monkeypatch.setattr(api.sentinel, "armed", lambda: False)
    host = SimpleNamespace(renew=lambda *a, **k: pytest.fail("unauthorized renewal"))
    daemon = api.Daemon(None, tmp_path, "fixture", workspaces=host)
    handler = object.__new__(api.make_handler(daemon))
    handler.path = f"/workspace/v1/projects/{PROJECT}/renew"
    handler.connection = object() if attack == "plain" else Mock(spec=ssl.SSLSocket)
    secret = derive_token(b"wrong-key" if attack == "wrong-key" else member.key)
    handler._sealing = lambda: (None, SimpleNamespace(secret=secret), attack != "unsealed", {})
    replies = []
    handler._send = lambda code, body: replies.append((code, body))
    assert handler._workspace(json.dumps({"cluster": member.group,
        "cluster_id": hashlib.sha256(member.key).hexdigest()}).encode())
    assert replies[0][0] == 403


def test_remote_renewal_uses_existing_capability_without_recovery(tmp_path, monkeypatch):
    remote = RemoteWorkspace.__new__(RemoteWorkspace)
    remote.base, remote.project_id = tmp_path / "private", PROJECT
    remote.mode, remote.cluster, remote.cluster_id = "dev", "fresh-dev", CURRENT
    remote.device_cert = "pinned"
    token = "mlws1.worker.existing"
    tokens.store(remote.base, "worker", token)
    requests = []
    def request(action, body):
        requests.append((action, body))
        return {"id": "worker", "project_id": PROJECT, "cluster_id": CURRENT}
    monkeypatch.setattr(remote, "_request", request)
    assert remote.renew("worker")["id"] == "worker"
    assert [(action, {key: value for key, value in body.items() if key != "device"}) for action, body in requests] == [
        ("renew", {"agent_token": token, "cluster": "fresh-dev", "cluster_id": CURRENT})]
    assert requests[0][1]["device"]["verification"] == "local-observed"
    assert tokens.load(remote.base, "worker") == token


@pytest.mark.parametrize("case", [("dev", "worker", "c" * 64, True),
    ("dev", "worker", CURRENT, False), ("prod", "worker", "c" * 64, False),
    ("dev", "other", "c" * 64, False), ("dev", "worker", "", True)])
def test_saved_connection_refresh_is_determined_by_cluster_scope(tmp_path, monkeypatch, case):
    mode, actor, old_id, renew = case
    key = b"new-key"
    new_id = hashlib.sha256(key).hexdigest()
    stored = new_id if old_id == CURRENT else old_id
    connection = {"agent": "worker", "local_agent": "local-worker", "cluster_id": stored,
                  "cluster": "old-dev", "host": "https://board.invalid", "project_id": PROJECT,
                  "root": str(tmp_path)}
    monkeypatch.setattr(automatic_connection, "memberships", lambda path: [SimpleNamespace(group="fresh-dev", key=key, mode=mode)])
    calls = []
    monkeypatch.setattr(automatic_connection, "RemoteWorkspace", lambda *a, **k: calls.append(k) or "remote")
    monkeypatch.setattr(automatic_connection, "bind", lambda *a, **k: {**connection, "cluster_id": new_id})
    result = automatic_connection.refresh(connection, actor)
    assert bool(calls) is renew
    assert result["cluster_id"] == (new_id if renew else stored)


def test_bind_renews_saved_scope_and_persists_current_cluster(tmp_path, monkeypatch):
    monkeypatch.setenv("POOLHOUSE_HOME", str(tmp_path / "state"))
    root = tmp_path / "checkout"
    root.mkdir()
    prior = {"agent": "worker", "host": "https://board.invalid", "project_id": PROJECT,
             "cluster_id": "c" * 64, "local_agent": "local-worker"}
    monkeypatch.setattr(project_connection, "selected", lambda path: prior)
    events = []
    remote = SimpleNamespace(host=prior["host"], project_id=PROJECT, cluster_key="", cluster_id=CURRENT,
                             mode="dev", token=lambda **kw: "existing")
    remote.renew = lambda agent: events.append(agent)
    remote.call = lambda *a: {"id": "worker", "role": "agent", "project": {"key": PROJECT, "cluster_id": CURRENT}}
    made = project_connection.bind(remote, root, "worker", "fresh-dev", local_agent="local-worker")
    assert events == ["worker"]
    assert made["cluster_id"] == CURRENT
    assert project_connection._saved()[str(root)]["cluster_id"] == CURRENT
