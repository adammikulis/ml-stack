"""Project-scoped agent authentication across a fleet transport."""

import socket
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_project_source import repository  # noqa: F401

from ml_stack import http
from ml_stack.fleet import tls
from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.daemon import ALL_INTERFACES
from ml_stack.fleet.discovery import Advertiser, Beacon, derive_token
from ml_stack.fleet.framing import LimitedServer
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.projects import ProjectRegistry
from ml_stack.fleet.remote import Peer
from ml_stack.http import Server, ServerError, request_json
from ml_stack.workspace import cli, project_connection, remote as remote_module, tokens
from ml_stack.workspace.identity import AGENT, HUMAN, LEAD
from ml_stack.workspace.remote import RemoteWorkspace
from ml_stack.workspace.remote_host import WorkspaceHost

PROJECT = "a" * 32
OTHER = "b" * 32


@pytest.mark.parametrize("discovered", [False, True])
def test_project_authority_routes_before_global_coordinator_discovery(monkeypatch, discovered):
    connection = {"host": "https://192.0.2.1", "project_id": PROJECT, "agent": "worker"}
    monkeypatch.setattr(project_connection, "selected", lambda *a: None if discovered else connection)
    monkeypatch.setattr(project_connection, "auto_attach", lambda *a: connection, raising=False)
    monkeypatch.setattr(cli.coordinator_config, "load", lambda *a: {})
    monkeypatch.setattr(cli.coordinator_client, "client", lambda *a: pytest.fail("global discovery"))
    monkeypatch.setattr(cli, "_context", lambda args, chosen=None: (chosen, "project-session"))
    args = SimpleNamespace(json=True)
    seen = []
    run = cli._runner(lambda args, ws, token: seen.append((ws, token)) or {"state": "connected"})
    assert run(args) == 0
    assert seen == [(connection, "project-session")]


@pytest.mark.parametrize("unavailable", [False, True])
def test_agent_connect_selects_explicit_project_instead_of_current_board(tmp_path, monkeypatch, unavailable):
    from ml_stack.workspace.identity import Denied
    requested = tmp_path / "other-project"
    current = {"project_id": "current-board"}
    target = {"project_id": "requested-board"}
    seen = []
    def selected(path=None):
        seen.append(path)
        if path == requested:
            if unavailable:
                raise Denied("requested project unavailable")
            return target
        return current
    monkeypatch.setattr(project_connection, "selected", selected)
    monkeypatch.setattr(cli.coordinator_client, "client", lambda *a: pytest.fail("global fallback"))
    canonical = SimpleNamespace(auth=lambda token: SimpleNamespace(id="worker"),
                                registry=SimpleNamespace(info=lambda name: {"project": {"key": "requested-board"}}))
    monkeypatch.setattr(cli, "_context", lambda args, connection: (canonical, "session")
                        if connection is target else pytest.fail("wrong board"))
    results = []
    monkeypatch.setattr(cli, "_show", lambda args, result: results.append(result))
    args = SimpleNamespace(agent="worker", project=str(requested), no_project=False,
                           one_agent=False, remote=False, code_only=False, name="", json=True)
    assert cli._bare(cli._connect)(args) == (3 if unavailable else 0)
    assert seen and all(path == requested for path in seen)
    assert results == ([] if unavailable else [{"id": "worker", "project": "requested-board", "state": "connected"}])


def test_local_runner_discovers_authority_once(monkeypatch):
    discoveries = []
    def discover():
        discoveries.append(True)
        assert len(discoveries) == 1
        return None
    class LocalWorkspace:
        registry = SimpleNamespace(_record_device=lambda *args: None)
        def auth(self, token):
            return SimpleNamespace(id="local-agent")
    monkeypatch.setattr(cli, "_project_connection", discover)
    monkeypatch.setattr(cli.coordinator_client, "client", lambda *a: None)
    monkeypatch.setattr(cli, "Workspace", LocalWorkspace)
    monkeypatch.setattr(cli, "_local_token", lambda args: "local-session")
    seen = []
    assert cli._runner(lambda args, ws, token: seen.append((ws, token)))(SimpleNamespace(json=True, cmd="whoami")) == 0
    assert len(seen) == 1 and isinstance(seen[0][0], LocalWorkspace)
    assert seen[0][1] == "local-session"
    assert discoveries == [True]


def test_coordinator_runner_keeps_selected_authority(monkeypatch):
    discoveries = []
    def discover():
        discoveries.append(True)
        assert len(discoveries) == 1
        return None
    calls = []
    remote = SimpleNamespace(command=lambda argv, token, request_id: calls.append(token))
    monkeypatch.setattr(cli, "_project_connection", discover)
    monkeypatch.setattr(cli.coordinator_client, "client", lambda *a: remote)
    monkeypatch.setattr(cli.coordinator_client, "argv_for", lambda *a: [])
    monkeypatch.setattr(cli, "_coordinator_token", lambda args, selected: "coordinator-session"
                        if selected is remote else pytest.fail("different coordinator"))
    args = SimpleNamespace(cmd="whoami", json=True, request_id="")
    assert cli._runner(lambda *a: pytest.fail("local dispatch"))(args) == 0
    assert calls == ["coordinator-session"]
    assert discoveries == [True]


def test_unavailable_selected_project_cannot_fall_back_to_global_authority(monkeypatch):
    from ml_stack.workspace.identity import Denied
    def selected(*args):
        raise Denied("selected project unavailable")
    monkeypatch.setattr(project_connection, "selected", selected)
    monkeypatch.setattr(cli.coordinator_client, "client", lambda *a: pytest.fail("authority fallback"))
    assert cli._runner(lambda *a: pytest.fail("unavailable project dispatch"))(SimpleNamespace(json=True)) == 3


def test_canonical_client_recovers_and_remembers_device_scoped_identity(tmp_path, monkeypatch):
    remote = RemoteWorkspace.__new__(RemoteWorkspace)
    remote.base, remote.project_id = tmp_path / "sessions", PROJECT
    requests = []
    transports = []
    monkeypatch.setattr(remote, "_device_transport", lambda: transports.append("device"))
    def request(action, payload):
        requests.append((action, payload))
        return {"id": "worker-peer", "token": "mlws1.worker-peer.saved", "project_id": PROJECT}
    monkeypatch.setattr(remote, "_request", request)
    monkeypatch.setattr(remote, "call", lambda operation, token: {"id": "worker-peer"})
    token = remote.token(agent="worker")
    assert requests[0][0] == "ensure"
    assert requests[0][1]["name"] == "worker"
    assert requests[0][1]["project"] == {"key": PROJECT}
    assert requests[0][1]["agent_token"] == ""
    assert remote.token(agent="worker-peer") == token
    assert len(requests) == 1 and len(transports) == 2
    (tokens.directory(remote.base) / "worker-peer").unlink()
    assert remote.token(agent="worker") == token
    assert requests[-1][1]["name"] == "worker"


@pytest.mark.parametrize("status, recovered", [(403, True), (503, False)])
def test_canonical_client_recovers_expiry_and_preserves_outage_failure(tmp_path, monkeypatch, status, recovered):
    remote = RemoteWorkspace.__new__(RemoteWorkspace)
    remote.base, remote.project_id = tmp_path / "sessions", PROJECT
    tokens.store(remote.base, "worker", "mlws1.worker.saved")
    requests = []
    monkeypatch.setattr(remote, "_device_transport", lambda: None)
    def call(operation, token):
        from ml_stack.workspace.identity import Denied
        raise Denied("unavailable") from ServerError("unavailable", status=status)
    monkeypatch.setattr(remote, "call", call)
    monkeypatch.setattr(remote, "_request", lambda action, payload:
                        requests.append(action) or {"id": "worker", "token": "mlws1.worker.recovered"})
    if recovered:
        assert remote.token(agent="worker") == "mlws1.worker.recovered"
        assert requests == ["ensure"]
    else:
        from ml_stack.workspace.identity import Denied
        with pytest.raises(Denied, match="unavailable"):
            remote.token(agent="worker")
        assert requests == []


@pytest.mark.parametrize("unsafe", ["is a symlink or Windows reparse point", "belongs to another user",
                                     "mode 644 lets others read it"])
def test_canonical_recovery_refuses_unsafe_credential_storage(tmp_path, monkeypatch, unsafe):
    remote = RemoteWorkspace.__new__(RemoteWorkspace)
    remote.base, remote.project_id = tmp_path / "sessions", PROJECT
    credential = tokens.store(remote.base, "worker", "mlws1.worker.saved")
    original = tokens.problem
    monkeypatch.setattr(tokens, "problem", lambda path: unsafe if path == credential else original(path))
    monkeypatch.setattr(remote, "_device_transport", lambda: None)
    monkeypatch.setattr(remote, "_request", lambda *a: pytest.fail("unsafe storage recovery"))
    from ml_stack.workspace.identity import Denied
    with pytest.raises(Denied, match="project session storage"):
        remote.token(agent="worker")


def test_canonical_recovery_serializes_read_ensure_and_store(tmp_path, monkeypatch):
    first, second = RemoteWorkspace.__new__(RemoteWorkspace), RemoteWorkspace.__new__(RemoteWorkspace)
    for remote in (first, second):
        remote.base, remote.project_id = tmp_path / "sessions", PROJECT
        monkeypatch.setattr(remote, "_device_transport", lambda: None)
    first_request, second_lock, release = threading.Event(), threading.Event(), threading.Event()
    original = remote_module.held
    @contextmanager
    def observed_lock(path):
        if threading.current_thread().name == "second-client":
            second_lock.set()
        with original(path):
            yield
    monkeypatch.setattr(remote_module, "held", observed_lock)
    count, results, errors = [], [], []
    def request(action, payload):
        count.append(action)
        secret = f"mlws1.worker.secret{len(count)}"
        if len(count) == 1:
            first_request.set()
            assert release.wait(5)
        return {"id": "worker", "token": secret}
    def call(operation, token):
        from ml_stack.workspace.identity import Denied
        if token != f"mlws1.worker.secret{len(count)}":
            raise Denied("stale credential") from ServerError("expired", status=403)
        return {"id": "worker"}
    for remote in (first, second):
        monkeypatch.setattr(remote, "_request", request)
        monkeypatch.setattr(remote, "call", call)
    def run(remote):
        try:
            results.append(remote.token(agent="worker"))
        except Exception as error:
            errors.append(error)
    workers = [threading.Thread(target=run, args=(first,), name="first-client"),
               threading.Thread(target=run, args=(second,), name="second-client")]
    workers[0].start()
    assert first_request.wait(5)
    workers[1].start()
    assert second_lock.wait(5)
    release.set()
    for worker in workers:
        worker.join(5)
        assert not worker.is_alive()
    assert not errors and results == ["mlws1.worker.secret1"] * 2
    assert count == ["ensure"]
    assert tokens.load(first.base, "worker") == results[0]


def test_canonical_first_use_accepts_directory_created_by_another_client(tmp_path, monkeypatch):
    remote = RemoteWorkspace.__new__(RemoteWorkspace)
    remote.base, remote.project_id = tmp_path / "sessions", PROJECT
    tokens.prepare(remote.base)
    existing = type(remote.base).exists
    monkeypatch.setattr(type(remote.base), "exists", lambda path: False if path == remote.base else existing(path))
    monkeypatch.setattr(remote, "_device_transport", lambda: None)
    monkeypatch.setattr(remote, "_request", lambda *a: {"id": "worker", "token": "mlws1.worker.saved"})
    assert remote.token(agent="worker") == "mlws1.worker.saved"


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
    ws.registry.set_project(ws.auth(owner), "privileged", {"key": PROJECT})
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
    daemon.workspaces = host
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
        project_root = tmp_path / "agent-project"
        project_root.mkdir()
        project_connection.bind(remote, project_root, "mac")
        monkeypatch.chdir(project_root)
        args = SimpleNamespace(agent="", token_file="")
        canonical, selected_token = cli._context(args)
        canonical.announce(selected_token, "joined", "Mac agent attached", "")
        assert "Mac agent attached" in host.status(PROJECT)["messages"][0]["text"]
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


def test_remote_client_never_reuses_global_or_human_token_files(monkeypatch, tmp_path):
    monkeypatch.setattr("ml_stack.workspace.remote.load_cluster_key", lambda path: bytes(range(32)))
    monkeypatch.setenv("ML_STACK_WORKSPACE_TOKEN", "global-human-token")
    monkeypatch.delenv(tokens.AGENT_ENV, raising=False)
    remote = RemoteWorkspace("http://127.0.0.1:8770", PROJECT)
    with pytest.raises(PermissionError, match="attach a project agent"):
        remote.token()
    owner = tmp_path / ".owner"
    owner.write_text("private-human-token")
    owner.chmod(0o600)
    with pytest.raises(PermissionError, match="only private project agent"):
        remote.token(token_file=str(owner))


def test_https_client_discovers_pins_and_authenticates_self_signed_host(host, tmp_path, monkeypatch):
    key = bytes(range(32))
    ident = tls.identity(tmp_path / "tls", "project-host")
    files = tmp_path / "tls-files"
    files.mkdir()
    runner = JobRunner(tmp_path / "tls-jobs", files)
    daemon = Daemon(runner, files, derive_token(key))
    daemon.projects = host.projects
    daemon.workspaces = host
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


def test_source_publication_does_not_initialize_board_and_explicit_invite_selects_authority(repository, tmp_path):  # noqa: F811
    registry = ProjectRegistry(tmp_path / "daemon", "pc", (repository,), host="https://192.168.2.59:8770")
    project = registry.share(registry.candidates()[0]["id"], "ml-stack")
    host = WorkspaceHost(registry)
    status = host.status(project.id)
    assert status["state"] == "unconfigured"
    assert not (registry.root / "shared-workspaces").exists()
    assert not project.authority_machine
    host.invite(project.id)
    assert project.authority_machine == "pc"
    assert project.board_host == "https://192.168.2.59:8770"
    assert host.status(project.id)["state"] == "awaiting_agents"


def test_foreign_canonical_authority_cannot_be_replaced_by_local_invite(repository, tmp_path):  # noqa: F811
    registry = ProjectRegistry(tmp_path / "daemon", "pc", (repository,), host="https://192.168.2.59:8770")
    project = registry.share(registry.candidates()[0]["id"], "ml-stack")
    project.authority_machine, project.board_host = "mac", "https://192.168.2.27:8770"
    host = WorkspaceHost(registry)
    assert host.status(project.id)["state"] == "connection_required"
    with pytest.raises(ValueError, match="another workspace authority"):
        host.invite(project.id)
    assert project.authority_machine == "mac"
