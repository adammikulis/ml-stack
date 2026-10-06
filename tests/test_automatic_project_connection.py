"""Authenticated same-project canonical Board selection."""

from types import SimpleNamespace

import pytest

from ml_stack.workspace import automatic_connection as automatic
from ml_stack.workspace.identity import Denied

PROJECT = "a" * 32


def peer(machine, authority="", host=""):
    address = f"https://{machine}:8770"
    row = {"id": PROJECT, "name": "sample", "machine": machine,
           "authority_machine": authority, "board_host": host}
    return SimpleNamespace(base_url=address, beacon=SimpleNamespace(machine=machine),
                           document={"machine": machine, "boards": [row]})


@pytest.fixture
def found(monkeypatch):
    nodes = []
    monkeypatch.setattr(automatic, "identity", lambda root: PROJECT)
    monkeypatch.setattr(automatic, "memberships", lambda path: [
        SimpleNamespace(group="development", key=b"key", mode="dev")])
    monkeypatch.setattr(automatic.Peer, "discover", lambda **kwargs: nodes)
    monkeypatch.setattr(automatic, "catalogue", lambda node, **kwargs: node.document)
    return nodes


def test_unconfigured_boards_choose_one_stable_device(found, tmp_path):
    found.extend([peer("node-b"), peer("node-a")])
    result = automatic.discover(tmp_path)
    assert result["host"] == "https://node-a:8770"
    assert result["authority_machine"] == "node-a"
    found.reverse()
    assert automatic.discover(tmp_path) == result


def test_existing_authority_is_preserved(found, tmp_path):
    found.extend([peer("node-a"), peer("node-b", "node-b", "https://node-b:8770")])
    assert automatic.discover(tmp_path)["authority_machine"] == "node-b"


@pytest.mark.redteam
def test_competing_authorities_are_refused(found, tmp_path):
    found.extend([peer("node-a", "node-a", "https://node-a:8770"),
                  peer("node-b", "node-b", "https://node-b:8770")])
    with pytest.raises(Denied, match="competing"):
        automatic.discover(tmp_path)


@pytest.mark.redteam
def test_advertised_authority_address_requires_matching_authenticated_peer(found, tmp_path):
    found.append(peer("node-a", "node-a", "https://node-b:8770"))
    with pytest.raises(Denied, match="address"):
        automatic.discover(tmp_path)


@pytest.mark.redteam
def test_catalogue_machine_cannot_spoof_authenticated_device(found, tmp_path):
    node = peer("node-a")
    node.document["machine"] = "node-b"
    found.append(node)
    with pytest.raises(Denied, match="authenticated device"):
        automatic.discover(tmp_path)


def test_prod_does_not_discover_automatic_boards(found, tmp_path, monkeypatch):
    monkeypatch.setattr(automatic, "memberships", lambda path: [
        SimpleNamespace(group="production", key=b"key", mode="prod")])
    found.append(peer("node-a"))
    assert automatic.discover(tmp_path) is None


def test_missing_selected_authority_does_not_elect_replacement(found, tmp_path):
    found.append(peer("node-a", "node-b", "https://node-b:8770"))
    with pytest.raises(Denied, match="unavailable"):
        automatic.discover(tmp_path)


def test_enrollment_binds_returned_agent_identity(found, tmp_path, monkeypatch):
    found.append(peer("node-a"))
    monkeypatch.setattr(automatic, "selected", lambda root: None)
    events = []
    class Remote:
        def __init__(self, host, project, **kwargs):
            self.host, self.project_id = host, project
        def enroll(self, name, **kwargs):
            events.append((name, kwargs))
            return {"id": "worker-1"}
    monkeypatch.setattr(automatic, "RemoteWorkspace", Remote)
    monkeypatch.setattr(automatic, "bind", lambda remote, root, agent, cluster: {
        "host": remote.host, "agent": agent, "cluster": cluster})
    result = automatic.connect(tmp_path, "worker", model="test-model", harness="codex")
    assert result["agent"] == "worker-1"
    assert events == [("worker", {"model": "test-model", "harness": "codex",
                                  "authority_machine": "node-a"})]


@pytest.mark.redteam
def test_discovery_bounds_authenticated_candidates(found, tmp_path):
    found.extend(peer(f"node-{number}") for number in range(automatic.MAX_PEERS + 1))
    with pytest.raises(Denied, match="device limit"):
        automatic.discover(tmp_path)


def test_discovery_refuses_elapsed_budget(found, tmp_path, monkeypatch):
    found.append(peer("node-a"))
    times = iter((0.0, automatic.DISCOVERY_SECONDS + 1))
    monkeypatch.setattr(automatic.time, "monotonic", lambda: next(times))
    with pytest.raises(Denied, match="time limit"):
        automatic.discover(tmp_path)
