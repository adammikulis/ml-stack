"""Canonical project CLI dispatch without device-local fallback."""

from types import SimpleNamespace

import pytest

from ml_stack import home
from ml_stack.files import write_json
from ml_stack.fleet import discovery, project_client, remote
from ml_stack.workspace import cli, project_connection as connection
from ml_stack.workspace.identity import Denied

PROJECT = "a" * 32


class Remote:
    host = "http://127.0.0.1:8770"
    project_id = PROJECT
    cluster_key = ""

    def __init__(self, *args, **kwargs):
        self.calls = []

    def token(self, **kwargs):
        return "project-agent-capability"

    def call(self, operation, token, *args, **kwargs):
        self.calls.append((operation, token, args, kwargs))
        if operation == "whoami":
            return {"id": "mac", "role": "agent", "can": ["send", "read", "claim"],
                    "project": {"key": PROJECT}}
        return {"seq": 1}


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
    root = tmp_path / "project"
    root.mkdir()
    return root


def test_normal_cli_sends_to_saved_canonical_board(project, monkeypatch):
    remote = Remote()
    connection.bind(remote, project, "mac", "default")
    monkeypatch.chdir(project)
    monkeypatch.setattr(connection, "RemoteWorkspace", lambda *a, **k: remote)
    monkeypatch.setattr(cli, "Workspace", lambda: pytest.fail("created a split local board"))
    args = SimpleNamespace(agent="", token_file="", label="helper", to="pc", type="status",
                           body="connected", subject="", reply_to=0, ttl=0, json=True)
    assert cli._runner(cli._send)(args) == 0
    operation, token, values, kwargs = remote.calls[-1]
    assert operation == "send" and token == "project-agent-capability"
    assert values == ("pc", "status", "connected") and kwargs["label"] == "helper"


def test_connected_project_uses_nearest_root_and_refuses_other_authority(project):
    connection.bind(Remote(), project, "mac")
    nested = project / "subdir"
    nested.mkdir()
    assert connection.selected(nested)["agent"] == "mac"
    other = Remote()
    other.host = "http://127.0.0.1:8771"
    with pytest.raises(Denied, match="another canonical board"):
        connection.bind(other, project, "mac")


def test_unconfigured_shared_checkout_never_uses_local_workspace(project):
    write_json(project / ".ml-stack-project.json", {"kind": "project-checkout", "project_id": PROJECT,
                                                   "authority": {}})
    with pytest.raises(Denied, match="no canonical board"):
        connection.selected(project)


def test_offline_canonical_connection_does_not_fall_back(project, monkeypatch):
    connection.bind(Remote(), project, "mac")
    monkeypatch.chdir(project)
    def offline(*args, **kwargs):
        raise Denied("canonical host offline")
    monkeypatch.setattr(connection, "RemoteWorkspace", offline)
    monkeypatch.setattr(cli, "Workspace", lambda: pytest.fail("created a split local board"))
    assert cli._runner(cli._agents)(SimpleNamespace(agent="", token_file="", json=True)) == 3


def test_corrupt_connection_record_disables_local_fallback(project):
    connection.bind(Remote(), project, "mac")
    home.state("workspace-connections.json").write_text("not json")
    with pytest.raises(Denied, match="local fallback is disabled"):
        connection.selected(project)


def test_unsupported_privileged_operation_is_explicitly_refused():
    ws = connection.CanonicalWorkspace(Remote(), "project-agent-capability")
    with pytest.raises(Denied, match="local fallback is disabled"):
        ws.mint("project-agent-capability", "lead", "lead")


def test_auto_discovery_selects_only_the_project_authority(monkeypatch):
    member = SimpleNamespace(group="home", key=b"cluster-key")
    host = SimpleNamespace(base_url="https://host.local:8770",
                           beacon=SimpleNamespace(machine="authority-machine"))
    source = SimpleNamespace(base_url="https://source.local:8770",
                             beacon=SimpleNamespace(machine="source-machine"))
    monkeypatch.setattr(discovery, "memberships", lambda: [member])
    monkeypatch.setattr(remote.Peer, "discover", classmethod(lambda cls, **kwargs: [host, source]))
    monkeypatch.setattr(project_client, "catalogue", lambda peer: {"projects": [
        {"id": PROJECT, "authority_machine": "authority-machine"}]})
    assert connection._find_authority(PROJECT) == ("home", host.base_url)


def test_auto_discovery_refuses_conflicting_project_authorities(monkeypatch):
    member = SimpleNamespace(group="home", key=b"cluster-key")
    hosts = [SimpleNamespace(base_url=f"https://host-{i}.local:8770",
                             beacon=SimpleNamespace(machine=f"machine-{i}")) for i in range(2)]
    monkeypatch.setattr(discovery, "memberships", lambda: [member])
    monkeypatch.setattr(remote.Peer, "discover", classmethod(lambda cls, **kwargs: hosts))
    monkeypatch.setattr(project_client, "catalogue", lambda peer: {"projects": [
        {"id": PROJECT, "authority_machine": peer.beacon.machine}]})
    with pytest.raises(Denied, match="conflicting workspace authorities"):
        connection._find_authority(PROJECT)


def test_auto_attach_discovers_authority_without_registering_a_device(monkeypatch, tmp_path):
    from ml_stack.fleet import projects

    monkeypatch.setattr(connection.git, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=str(tmp_path)))
    monkeypatch.setattr(projects, "identity", lambda root: PROJECT)
    monkeypatch.setattr(connection, "_find_authority", lambda project: ("home", "https://host.local:8770"))
    monkeypatch.setattr(connection, "RemoteWorkspace", lambda host, project_id, **kwargs:
                        SimpleNamespace(host=host, cluster_key=""))
    assert connection.auto_attach(tmp_path) == {
        "host": "https://host.local:8770", "project_id": PROJECT, "agent": "",
        "cluster": "home", "cluster_key": "", "root": str(tmp_path.resolve())}
