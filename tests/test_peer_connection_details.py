"""Peer-facing connection endpoints and pinned device checks."""

from types import SimpleNamespace

import pytest

from ml_stack.fleet import project_board_routes, projects
from ml_stack.workspace import connection_details
from ml_stack.workspace.identity import Denied


@pytest.mark.parametrize("address", ["", "127.0.0.1", "0.0.0.0", "224.1.2.3", "8.8.8.8", "::1"])  # noqa: S104
def test_shared_host_refuses_non_network_addresses(monkeypatch, address):
    monkeypatch.delenv("ML_STACK_WSL_NETWORK", raising=False)
    monkeypatch.setattr(projects, "primary_ip", lambda: address)
    assert projects.lan_host(8770) == ""


@pytest.mark.parametrize("address, expected", [("192.168.40.2", "192.168.40.2"),
                                               ("fd00::2", "[fd00::2]")])
def test_shared_host_uses_current_route_address(monkeypatch, address, expected):
    monkeypatch.delenv("ML_STACK_WSL_NETWORK", raising=False)
    monkeypatch.delenv("ML_STACK_FLEET_TLS", raising=False)
    monkeypatch.setattr(projects, "primary_ip", lambda: address)
    assert projects.lan_host(8770) == f"https://{expected}:8770"


@pytest.fixture
def connection(monkeypatch):
    identity = {"id": "worker", "project": {"key": "a" * 32}, "role": "agent"}
    remote = SimpleNamespace(host="https://127.0.0.1:8770", project_id="a" * 32,
                             authority_machine="device", device_cert="Y2VydA==",
                             cluster="dev", cluster_key="", call=lambda *args: identity)
    calls = []
    def candidate(host, project_id, **kwargs):
        calls.append((host, project_id, kwargs))
        return remote
    monkeypatch.setattr(connection_details, "RemoteWorkspace", candidate)
    monkeypatch.setattr(connection_details.home, "machine_id", lambda: "device")
    monkeypatch.setattr(connection_details, "lan_host", lambda port: "https://192.168.40.2:8770")
    return remote, calls


def test_connection_exports_lan_and_preserves_local_namespace(connection):
    remote, calls = connection
    result = connection_details.details(remote, "private-fixture")
    assert result["host"] == "https://192.168.40.2:8770"
    assert result["authority_machine"] == "device"
    assert result["agent"] == "worker"
    assert result["state"] == "verified"
    assert remote.host == "https://127.0.0.1:8770"
    assert len(calls) == 1
    assert "private-fixture" not in str(result)


def test_offline_connection_does_not_export_loopback(connection, monkeypatch):
    remote, calls = connection
    monkeypatch.setattr(connection_details, "lan_host", lambda port: "")
    with pytest.raises(Denied, match="no reachable network address"):
        connection_details.details(remote, "private-fixture")
    assert calls == []


def test_swapped_network_identity_refused_before_capability_transmission(connection, monkeypatch):
    remote, _ = connection
    candidate = SimpleNamespace(authority_machine="foreign", device_cert=remote.device_cert,
                                call=lambda *args: pytest.fail("capability transmitted"))
    monkeypatch.setattr(connection_details, "RemoteWorkspace", lambda *args, **kwargs: candidate)
    with pytest.raises(Denied, match="project authority"):
        connection_details.details(remote, "private-fixture")


@pytest.mark.parametrize("network_host, status", [("https://192.168.40.2:8770", 201), ("", 409)])
def test_copied_invitation_uses_current_network_before_creating_code(monkeypatch, network_host, status):
    invites, responses = [], []
    project = SimpleNamespace(authority_machine="device", board_host="https://127.0.0.1:8770")
    class Route(project_board_routes.ProjectBoardRoutes):
        path = "/ui/projects/" + "a" * 32 + "/invite"
        client_ip, cookie, host_header, method = "127.0.0.1", "session", "localhost:8770", "POST"
        ui = SimpleNamespace(projects=SimpleNamespace(machine="device", get=lambda _: project),
                             workspaces=SimpleNamespace(invite=lambda *args: invites.append(args) or {"code": "fixture"}),
                             peer_port=8770, cluster_key_path=None, authed=lambda _: True, host_ok=lambda _: True)
        def header(self, key, default=""):
            return "http://localhost:8770" if key == "Origin" else default
        def body(self):
            return {"hint": "worker"}
        def send(self, code, result):
            responses.append((code, result))
    monkeypatch.setattr(project_board_routes, "lan_host", lambda _: network_host)
    monkeypatch.setattr(project_board_routes, "memberships", lambda _: [SimpleNamespace(group="dev")])
    assert Route().route()
    assert responses[0][0] == status
    if network_host:
        assert network_host in responses[0][1]["command"]
        assert "127.0.0.1" not in responses[0][1]["command"]
        assert responses[0][1]["authority_machine"] == "device"
    else:
        assert not invites
