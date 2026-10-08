"""Authenticated same-project canonical Board selection."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ml_stack import home, sealing
from ml_stack.fleet import project_client, remote as fleet_remote
from ml_stack.http import Sealed
from ml_stack.workspace import automatic_connection as automatic
from ml_stack.workspace.identity import Denied

PROJECT = "a" * 32


def peer(machine, host=""):
    address = f"https://{machine}:8770"
    row = {"id": PROJECT, "name": "sample", "machine": machine,
           "board_host": host}
    return SimpleNamespace(base_url=address, beacon=SimpleNamespace(machine=machine, cert="fixture"),
                           document={"machine": machine, "boards": [row]})


@pytest.fixture
def found(monkeypatch):
    nodes = []
    monkeypatch.setattr(home, "machine_id", lambda: "node-a")
    monkeypatch.setattr(automatic, "identity", lambda root: PROJECT)
    monkeypatch.setattr(automatic, "_register", lambda *args: None)
    monkeypatch.setattr(automatic, "memberships", lambda path: [
        SimpleNamespace(group="development", key=b"key", mode="dev")])
    monkeypatch.setattr(automatic, "settle", lambda member, *args: member)
    monkeypatch.setattr(automatic.Peer, "discover", lambda **kwargs: nodes)
    monkeypatch.setattr(automatic, "catalogue", lambda node, **kwargs: node.document)
    return nodes


def test_unconfigured_boards_choose_one_stable_device(found, tmp_path):
    found.extend([peer("node-b"), peer("node-a")])
    result = automatic.discover(tmp_path)
    assert result["host"] == "https://node-a:8770"
    found.reverse()
    assert automatic.discover(tmp_path) == result


def test_existing_authority_is_preserved(found, tmp_path):
    found.extend([peer("node-a"), peer("node-b", "https://node-b:8770")])
    assert automatic.discover(tmp_path)["host"] == "https://node-b:8770"


@pytest.mark.redteam
def test_competing_authorities_are_refused(found, tmp_path):
    found.extend([peer("node-a", "https://node-a:8770"),
                  peer("node-b", "https://node-b:8770")])
    with pytest.raises(Denied, match="competing"):
        automatic.discover(tmp_path)


@pytest.mark.redteam
def test_advertised_authority_address_requires_matching_authenticated_peer(found, tmp_path):
    found.append(peer("node-a", "https://node-b:8770"))
    with pytest.raises(Denied, match="address"):
        automatic.discover(tmp_path)


def test_local_authority_accepts_its_lan_origin_from_loopback_discovery(found, tmp_path, monkeypatch):
    node = peer("node-a", "https://192.168.40.2:8770")
    node.base_url = "https://127.0.0.1:8770"
    found.append(node)
    monkeypatch.setattr(fleet_remote, "primary_ip", lambda: "192.168.40.2")
    assert automatic.discover(tmp_path)["host"] == node.base_url


@pytest.mark.parametrize("origin", ["https://192.168.40.3:8770", "https://192.168.40.2:8771",
                                    "http://192.168.40.2:8770", "https://192.168.40.2:8770/other",
                                    "https://user@192.168.40.2:8770"])
def test_local_origin_alias_requires_same_device_and_tls_port(found, tmp_path, monkeypatch, origin):
    node = peer("node-a", origin)
    node.base_url = "https://127.0.0.1:8770"
    found.append(node)
    monkeypatch.setattr(fleet_remote, "primary_ip", lambda: "192.168.40.2")
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
    found.append(peer("node-a", "https://node-b:8770"))
    with pytest.raises(Denied, match="exactly one"):
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
    monkeypatch.setattr(automatic, "bind", lambda remote, root, agent, cluster, **kwargs: {
        "host": remote.host, "agent": agent, "cluster": cluster})
    result = automatic.connect(tmp_path, "worker", claim=("test-model", "codex"))
    assert result["agent"] == "worker-1"
    assert events == [("worker", {"model": "test-model", "harness": "codex"})]


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


@pytest.fixture
def startup_remote(found, tmp_path, monkeypatch):
    found.append(peer("node-a"))
    monkeypatch.setattr(automatic.worktreerules, "checkouts", lambda root: (root, root / ".git"))
    monkeypatch.setattr(automatic, "selected", lambda root: None)
    monkeypatch.setattr(automatic.person, "marked", lambda: "")
    events = []
    class Remote:
        def __init__(self, host, project, **kwargs):
            self.host, self.project_id, self.base = host, project, tmp_path / "private"
        def enroll(self, name, *, model, harness):
            events.append(("enroll", name, model, harness))
            return {"id": "native-parent"}
        def delegate(self, parent, name):
            events.append(("delegate", parent, name))
            return {"id": f"{parent}/{name}"}
        def token(self, *, agent):
            events.append(("token", agent))
            return "private-test-capability"
        def call(self, operation, token):
            events.append((operation, token))
            return {"id": "connected-parent", "role": "agent"}
    monkeypatch.setattr(automatic, "RemoteWorkspace", Remote)
    monkeypatch.setattr(automatic, "bind", lambda remote, root, agent, cluster: events.append(("bind", agent)))
    return events


def test_native_person_start_creates_durable_parent_and_revocable_child(startup_remote, tmp_path):
    seat = automatic.startup(tmp_path, "worker", claim=("test-model", "codex"))
    assert seat.name == "native-parent/worker"
    assert seat.remote is not None
    assert seat.lifecycle_base == seat.remote.base
    assert startup_remote == [("enroll", "native-worker", "test-model", "codex"),
                              ("bind", "native-parent"),
                              ("delegate", "native-parent", "worker")]


@pytest.mark.redteam
def test_native_agent_start_cannot_mint_an_unrelated_project_parent(startup_remote, tmp_path, monkeypatch):
    monkeypatch.setattr(automatic.person, "marked", lambda: "ML_STACK_AGENT")
    with pytest.raises(Denied, match=r"parent.*connection"):
        automatic.startup(tmp_path, "worker", "local-parent")
    assert startup_remote == []


def test_native_agent_start_delegates_from_its_authenticated_project_parent(startup_remote, tmp_path, monkeypatch):
    monkeypatch.setattr(automatic.person, "marked", lambda: "ML_STACK_AGENT")
    monkeypatch.setattr(automatic, "selected", lambda root: {
        "host": "https://node-a:8770", "project_id": PROJECT,
        "cluster": "development", "agent": "connected-parent"})
    seat = automatic.startup(tmp_path, "worker", "connected-parent")
    assert seat.name == "connected-parent/worker"
    assert startup_remote[-1] == ("delegate", "connected-parent", "worker")
    assert not any(event[0] == "enroll" for event in startup_remote)


@pytest.mark.redteam
def test_native_agent_cannot_select_another_connected_parent(startup_remote, tmp_path, monkeypatch):
    monkeypatch.setattr(automatic.person, "marked", lambda: "ML_STACK_AGENT")
    monkeypatch.setattr(automatic, "selected", lambda root: {
        "host": "https://node-a:8770", "project_id": PROJECT,
        "cluster": "development", "agent": "connected-parent"})
    with pytest.raises(Denied, match="parent does not match"):
        automatic.startup(tmp_path, "worker", "foreign-parent")
    assert startup_remote == []


@pytest.mark.redteam
@pytest.mark.parametrize("attack", ["oversized", "unsealed", "tampered", "invalid-json", "wrong-project", "redirect"])
def test_local_registration_refuses_hostile_response(tmp_path, monkeypatch, attack):
    endpoint = f"http://127.0.0.1:{automatic.HTTP_PORT}/workspace/v1/local-project"
    sealed = Sealed(b"s" * 32, "registration-nonce")
    plain = b"not JSON" if attack == "invalid-json" else b'{"id":"' + PROJECT.encode() + b'"}'
    if attack == "wrong-project":
        plain = b'{"id":"' + b"b" * 32 + b'"}'
    raw = sealing.seal(sealed.key, plain, sealing.response_data(sealed.nonce, 200))
    if attack == "tampered":
        raw = raw[:-1] + bytes([raw[-1] ^ 1])
    if attack == "oversized":
        raw = b"x" * (65536 + sealing.NONCE_BYTES + 17)
    response = SimpleNamespace(status=200, headers={} if attack == "unsealed" else {sealing.HEADER: "2"},
                               sealed=sealed, read=lambda limit: raw[:limit])
    calls = []

    def transport(url, **options):
        calls.append((url, options))
        assert options["guard"](url) == endpoint
        assert options["timeout"] == 2
        assert options["headers"][sealing.HEADER] == "2"
        if attack == "redirect":
            options["guard"]("https://foreign.invalid/workspace/v1/local-project")
        return nullcontext(response)

    monkeypatch.setattr(project_client, "open_stream", transport)
    with pytest.raises((Denied, ValueError)):
        automatic._register(tmp_path, SimpleNamespace(key=b"fixture-dev-key"), PROJECT)
    assert len(calls) == 1 and calls[0][0] == endpoint


def test_secondary_dev_membership_cannot_override_active_prod(monkeypatch):
    monkeypatch.setattr(automatic, 'memberships', lambda path: [
        SimpleNamespace(group='production', mode='prod'), SimpleNamespace(group='development', mode='dev')])
    assert not automatic._active_dev({'cluster': 'development'})
    assert not automatic._active_dev({'cluster': 'production'})


@pytest.mark.parametrize("stored", [{}, {"authority_machine": "retired-device"}])
def test_saved_selection_without_an_authority_machine_attaches(tmp_path, monkeypatch, stored):
    selection = {"host": "https://node-a:8770", "project_id": PROJECT, "agent": "",
                 "cluster": "development", "cluster_key": "", **stored}
    monkeypatch.setattr(automatic, "selected", lambda root: selection)
    events = []
    class Remote:
        def __init__(self, host, project, **kwargs):
            self.host, self.project_id = host, project
        def enroll(self, name, **kwargs):
            events.append((name, kwargs))
            return {"id": name}
    monkeypatch.setattr(automatic, "RemoteWorkspace", Remote)
    monkeypatch.setattr(automatic, "bind", lambda remote, root, agent, cluster, **kwargs: {"agent": agent})
    assert automatic.attach(tmp_path, "claude", selection) == {"agent": "claude"}
    assert events == [("claude", {"model": "", "harness": ""})]
