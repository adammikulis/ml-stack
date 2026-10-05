"""Project-scoped agent authentication across a fleet transport."""

import socket
import threading
from types import SimpleNamespace

import pytest

from ml_stack import http
from ml_stack.fleet import tls
from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.daemon import ALL_INTERFACES
from ml_stack.fleet.discovery import Advertiser, Beacon, derive_token
from ml_stack.fleet.framing import LimitedServer
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.remote import Peer
from ml_stack.http import Server, ServerError, request_json
from ml_stack.workspace import tokens
from ml_stack.workspace.identity import AGENT, HUMAN, LEAD
from ml_stack.workspace.remote import RemoteWorkspace
from ml_stack.workspace.remote_host import WorkspaceHost

PROJECT = "a" * 32
OTHER = "b" * 32


@pytest.fixture
def host(tmp_path):
    class Projects:
        def get(self, project_id):
            if project_id not in {PROJECT, OTHER}:
                raise ValueError("unknown project")
            return SimpleNamespace(name="ml-stack")
        def workspace_base(self, project_id):
            self.get(project_id)
            return tmp_path / project_id
    made = WorkspaceHost(Projects())
    made.prepare(PROJECT)
    made.prepare(OTHER)
    return made


def joined(host, project=PROJECT, name="codex"):
    invite = host.invite(project)
    code, result = host.answer(project, "join", {"name": name, "code": invite["code"]})
    assert code == 201, result
    return result


def call(host, agent, operation, *args, project=PROJECT, **kwargs):
    return host.answer(project, "board", {"agent_token": agent["token"],
                       "operation": operation, "args": list(args), "kwargs": kwargs})


def test_same_project_agents_exchange_fenced_messages(host):
    first, second = joined(host, name="pc"), joined(host, name="mac")
    code, sent = call(host, first, "send", "mac", "status", "connected across the LAN")
    assert code == 200, sent
    code, inbox = call(host, second, "inbox")
    assert code == 200
    assert inbox["result"][0]["from"] == "pc"
    assert "connected across the LAN" in inbox["result"][0]["text"]
    assert inbox["result"][0]["trust"] != "authority"


@pytest.mark.parametrize("role", [HUMAN, LEAD])
def test_remote_human_and_lead_tokens_are_refused(host, role):
    ws = host.workspace(PROJECT)
    owner = tokens.read_file(tokens.directory(ws.base) / tokens.OWNER_FILE)
    token = ws.mint(owner, "privileged", role)
    code, _ = call(host, {"token": token}, "agents")
    assert code == 403


def test_capability_and_invite_cannot_cross_projects(host):
    agent = joined(host)
    assert call(host, agent, "agents", project=OTHER)[0] == 403
    invite = host.invite(PROJECT)
    assert host.answer(OTHER, "join", {"name": "wrong", "code": invite["code"]})[0] == 403


@pytest.mark.parametrize("operation", ["init", "mint", "registry.info", "files.read", "delegate",
                                       "quarantine_release", "__class__", "status"])
def test_operation_allowlist_blocks_privilege_and_unrelated_data(host, operation):
    assert call(host, joined(host), operation)[0] == 403


def test_unscoped_agent_token_is_refused(host):
    ws = host.workspace(PROJECT)
    owner = tokens.read_file(tokens.directory(ws.base) / tokens.OWNER_FILE)
    unscoped = ws.mint(owner, "unscoped", AGENT)
    assert call(host, {"token": unscoped}, "agents")[0] == 403


def test_remote_claim_does_not_use_a_foreign_process_id(host):
    agent = joined(host)
    code, result = call(host, agent, "claim", "branch", "feature", pid=99999999)
    assert code == 200, result
    assert host.workspace(PROJECT).claims.listing()[0]["pid"] == 0


def test_expired_or_reused_invite_is_refused(host):
    invite = host.invite(PROJECT)
    body = {"name": "first", "code": invite["code"]}
    assert host.answer(PROJECT, "join", body)[0] == 201
    assert host.answer(PROJECT, "join", {**body, "name": "second"})[0] == 403


def test_registered_agent_is_not_online_without_authenticated_remote_contact(host):
    ws = host.workspace(PROJECT)
    owner = tokens.read_file(tokens.directory(ws.base) / tokens.OWNER_FILE)
    ws.mint(owner, "local-only", AGENT)
    status = host.status(PROJECT)
    assert status["state"] == "offline"
    assert status["agents"][0]["online"] is False
    joined(host, name="mac")
    status = host.status(PROJECT)
    assert status["state"] == "connected"
    assert next(a for a in status["agents"] if a["id"] == "mac")["online"] is True


def test_status_includes_only_selected_project_board_messages(host):
    agent = joined(host)
    boards = call(host, agent, "board.list")[1]["result"]
    project_board = next(b["name"] for b in boards if b["project"])
    call(host, agent, "send", project_board, "note", "selected project update")
    call(host, agent, "send", "#general", "note", "not the selected board")
    messages = host.status(PROJECT)["messages"]
    assert len(messages) == 1
    assert "selected project update" in messages[0]["text"]


def test_signed_sealed_fleet_and_agent_capabilities_both_required(host, tmp_path, monkeypatch):
    key = bytes(range(32))
    fleet_token = derive_token(key)
    files = tmp_path / "files"
    files.mkdir()
    runner = JobRunner(tmp_path / "jobs", files)
    daemon = Daemon(runner, files, fleet_token)
    daemon.projects = host.projects
    server = Server(("127.0.0.1", 0), make_handler(daemon))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr("ml_stack.workspace.remote.load_cluster_key", lambda path: key)
    try:
        remote = RemoteWorkspace(base, PROJECT)
        invite = host.invite(PROJECT)
        connected = remote.join(invite["code"], "mac")
        assert "token" not in connected
        token = remote.token(agent="mac")
        assert remote.call("whoami", token)["id"] == "mac"
        with pytest.raises(ServerError):
            request_json(f"{base}/workspace/v1/projects/{PROJECT}/board",
                         payload={"agent_token": token, "operation": "agents"})
        with pytest.raises(PermissionError):
            remote.call("agents", "not-an-agent-capability")
    finally:
        server.shutdown()
        server.server_close()
        runner.shutdown()


def test_client_refuses_unsealed_response(monkeypatch):
    class PlainResponse:
        def __init__(self):
            self.headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
    monkeypatch.setattr("ml_stack.workspace.remote.load_cluster_key", lambda path: bytes(range(32)))
    monkeypatch.setattr("ml_stack.workspace.remote.open_stream", lambda *a, **k: PlainResponse())
    remote = RemoteWorkspace("http://127.0.0.1:8770", PROJECT)
    with pytest.raises(PermissionError, match="not authenticated and sealed"):
        remote.call("agents", "agent-capability")


def test_https_client_discovers_pins_and_authenticates_self_signed_host(host, tmp_path, monkeypatch):
    key = bytes(range(32))
    ident = tls.identity(tmp_path / "tls", "project-host")
    files = tmp_path / "tls-files"
    files.mkdir()
    runner = JobRunner(tmp_path / "tls-jobs", files)
    daemon = Daemon(runner, files, derive_token(key))
    daemon.projects = host.projects
    server = LimitedServer((ALL_INTERFACES, 0), make_handler(daemon), tls=tls.server_context(ident))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("", 0))
        udp = probe.getsockname()[1]
    advertiser = Advertiser(Beacon(name="project-host", port=server.server_port, cert=ident.beacon),
                            key, port=udp, interval_s=30).start()
    discover = Peer.discover
    monkeypatch.setattr("ml_stack.workspace.remote.load_cluster_key", lambda path: key)
    monkeypatch.setattr(Peer, "discover", lambda **kw: discover(key=kw["key"], port=udp, timeout_s=1))
    http._PINNED.clear()
    try:
        remote = RemoteWorkspace(f"https://127.0.0.1:{server.server_port}", PROJECT)
        assert f"127.0.0.1:{server.server_port}" in http._PINNED
        invite = host.invite(PROJECT)
        result = remote.join(invite["code"], "tls-mac")
        assert remote.call("whoami", remote.token(agent=result["id"]))["id"] == "tls-mac"
    finally:
        advertiser.stop()
        server.shutdown()
        server.server_close()
        runner.shutdown()
        http._PINNED.clear()


def test_adopted_history_has_no_identity_or_role_grants(host):
    agent = joined(host)
    before = host.workspace(PROJECT).registry.ids()
    result = host.adopt(PROJECT, {"board": "#ml-stack", "tokens": ["private-secret"],
                      "messages": [{"seq": 1, "from": "old-owner", "from_role": "human",
                                    "text": "old project message", "can": ["mint"]}]})
    assert result["messages"] == 1
    assert host.workspace(PROJECT).registry.ids() == before
    code, reply = call(host, agent, "history")
    assert code == 200
    row = reply["result"][0]
    assert row["authority"] == "none"
    assert "from_role" not in row and "can" not in row and "tokens" not in row
    assert "old project message" in row["text"]
    assert host.status(PROJECT)["history"] == reply["result"]


def test_history_from_other_boards_and_conflicting_imports_are_refused(host):
    with pytest.raises(ValueError, match="only the selected"):
        host.adopt(PROJECT, {"board": "#ml-stack", "messages": [{"board": "#other", "text": "x"}]})
    history = {"board": "#ml-stack", "messages": [{"text": "selected"}]}
    assert host.adopt(PROJECT, history) == host.adopt(PROJECT, history)
    with pytest.raises(ValueError, match="different adopted history"):
        host.adopt(PROJECT, {"board": "#ml-stack", "messages": [{"text": "different"}]})
