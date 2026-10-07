"""Automatic Dev project attachment through ordinary workspace commands."""

import hashlib
import socket
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from ml_stack.workspace import automatic_connection, cli, project_connection
from ml_stack.workspace.identity import Denied

pytest_plugins = ["test_project_source"]


@pytest.mark.parametrize("config", [{}, {"mode": "host"}])
def test_cli_discovers_unconfigured_dev_board_before_legacy_attachment(monkeypatch, config):
    choice = {"automatic": True, "host": "https://board.invalid", "project_id": "a" * 32}
    monkeypatch.setattr(project_connection, "selected", lambda cwd: None)
    monkeypatch.setattr(cli.coordinator_config, "load", lambda base: config)
    monkeypatch.setattr(automatic_connection, "local_project", lambda cwd: choice)
    monkeypatch.setattr(project_connection, "auto_attach", lambda cwd: pytest.fail("legacy discovery"))
    assert cli._project_connection() is choice


def test_remote_coordinator_discovers_dev_project_board(monkeypatch):
    monkeypatch.setattr(project_connection, "selected", lambda cwd: None)
    monkeypatch.setattr(cli.coordinator_config, "load", lambda base: {"mode": "remote"})
    choice = {"host": "https://board.invalid", "automatic": True}
    monkeypatch.setattr(automatic_connection, "local_project", lambda cwd: choice)
    monkeypatch.setattr(project_connection, "auto_attach", lambda cwd: pytest.fail("legacy discovery"))
    assert cli._project_connection() is choice


@pytest.mark.parametrize("model,harness", [("fixture-model", "codex"), ("", "")])
def test_cli_enrolls_authenticated_local_agent_without_manual_metadata(monkeypatch, tmp_path, model, harness):
    choice = {"automatic": True, "host": "https://board.invalid", "project_id": "a" * 32,
              "root": str(tmp_path), "agent": ""}
    local = SimpleNamespace(auth=lambda token: SimpleNamespace(id="worker", role="agent"),
                            registry=SimpleNamespace(info=lambda actor: {"model": model, "harness": harness}))
    monkeypatch.setattr(cli, "Workspace", lambda: local)
    monkeypatch.setattr(cli, "_local_token", lambda args: "authenticated-local-token")
    monkeypatch.setattr(automatic_connection.device_agent, "owned_project_session", lambda ws, token, *args: nullcontext(ws.auth(token)))
    enrolled = []
    def attach(root, name, choice, *, claim):
        enrolled.append((root, name, claim, choice))
        return {**choice, "agent": "worker-device"}
    monkeypatch.setattr(automatic_connection, "attach", attach)
    remote = SimpleNamespace(token=lambda **kw: "project-token")
    monkeypatch.setattr(project_connection, "RemoteWorkspace", lambda *a, **kw: remote)
    monkeypatch.setattr(project_connection, "CanonicalWorkspace", lambda remote, token: (remote, token))
    args = SimpleNamespace(agent="worker", token_file="")
    canonical, token = cli._context(args, choice)
    assert enrolled == [(tmp_path, "worker", (model, harness), choice)]
    assert canonical == (remote, token) and token == "project-token"
    assert args.agent == "worker-device"


@pytest.mark.parametrize("role,name", [("human", "owner"), ("lead", "lead"), ("agent", "worker/child")])
def test_cli_cannot_enroll_privileged_or_child_identity(monkeypatch, tmp_path, role, name):
    local = SimpleNamespace(auth=lambda token: SimpleNamespace(id=name, role=role))
    monkeypatch.setattr(cli, "Workspace", lambda: local)
    monkeypatch.setattr(cli, "_local_token", lambda args: "local-token")
    def owned(ws, token, *args):
        raise Denied("automatic project connection requires an authenticated local parent agent")
    monkeypatch.setattr(automatic_connection.device_agent, "owned_project_session", owned)
    monkeypatch.setattr(automatic_connection, "attach", lambda *a, **kw: pytest.fail("enrolled foreign role"))
    with pytest.raises(Denied, match="local parent agent"):
        cli._context(SimpleNamespace(agent=name, token_file=""), {"automatic": True, "root": str(tmp_path)})


def test_local_project_discovery_uses_checkout_root(monkeypatch, tmp_path):
    root = tmp_path / "project"
    nested = root / "nested"
    monkeypatch.setattr(automatic_connection.worktreerules, "checkouts", lambda cwd: (root, root))
    observed = []
    monkeypatch.setattr(automatic_connection, "discover", lambda cwd: observed.append(cwd) or {"host": "https://board.invalid"})
    choice = automatic_connection.local_project(nested)
    assert observed == [root]
    assert choice["root"] == str(root) and choice["automatic"] is True


def test_board_registration_waits_until_visible_singletons_converge(monkeypatch, tmp_path):
    member = SimpleNamespace(key=b"winner", group="development", mode="dev", selection="automatic")
    identity = hashlib.sha256(member.key).hexdigest()
    monkeypatch.setattr(automatic_connection, "memberships", lambda path: [member])
    monkeypatch.setattr(automatic_connection, "identity", lambda root: "a" * 32)
    monkeypatch.setattr(automatic_connection.automatic_clusters, "ensure", lambda *a, **kw: member)
    snapshots = iter([[('', {"cluster_id": 'old-cluster'})], [('', {"cluster_id": identity})]])
    checks = []
    def offers(port):
        checks.append(True)
        return next(snapshots)
    monkeypatch.setattr(automatic_connection.automatic_clusters, "offers", offers)
    monkeypatch.setattr(automatic_connection.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(automatic_connection, "_register", lambda *a: checks.append("registered"))
    monkeypatch.setattr(automatic_connection.Peer, "discover", lambda **kw: [])
    assert automatic_connection.discover(tmp_path) is None
    assert checks == [True, True, "registered"]


def test_unsettled_devices_never_register_or_claim_board(monkeypatch, tmp_path):
    member = SimpleNamespace(key=b"winner", group="development", mode="dev", selection="automatic")
    monkeypatch.setattr(automatic_connection, "memberships", lambda path: [member])
    monkeypatch.setattr(automatic_connection, "identity", lambda root: "a" * 32)
    monkeypatch.setattr(automatic_connection.automatic_clusters, "ensure", lambda *a, **kw: member)
    monkeypatch.setattr(automatic_connection.automatic_clusters, "offers", lambda port: [('', {"cluster_id": 'other'})])
    monkeypatch.setattr(automatic_connection, "ADMISSION_SECONDS", 0)
    monkeypatch.setattr(automatic_connection, "_register", lambda *a: pytest.fail("unsettled registration"))
    with pytest.raises(Denied, match="converged"):
        automatic_connection.discover(tmp_path)


@pytest.mark.parametrize("environment", [False, True])
def test_saved_connection_resolves_authenticated_local_alias(monkeypatch, tmp_path, environment):
    choice = {"host": "https://board.invalid", "project_id": "a" * 32, "root": str(tmp_path),
              "agent": "worker-device", "local_agent": "worker"}
    actor = SimpleNamespace(id="worker")
    monkeypatch.setattr(cli, "Workspace", lambda: object())
    monkeypatch.setattr(cli, "_local_token", lambda args: "local-token")
    checked = []
    monkeypatch.setattr(automatic_connection.device_agent, "owned_project_session", lambda ws, token, *args: nullcontext(checked.append(token) or actor))
    monkeypatch.setenv(cli.tokens.AGENT_ENV, "worker" if environment else "")
    used = []
    remote = SimpleNamespace(token=lambda **kw: used.append(kw) or "project-token")
    monkeypatch.setattr(project_connection, "RemoteWorkspace", lambda *a, **kw: remote)
    monkeypatch.setattr(project_connection, "CanonicalWorkspace", lambda *a: object())
    args = SimpleNamespace(agent="" if environment else "worker", token_file="")
    cli._context(args, choice)
    assert checked == ["local-token"]
    assert used == [{"agent": "worker-device", "token_file": ""}]


def test_saved_alias_revocation_never_reenrolls(monkeypatch, tmp_path):
    choice = {"host": "https://board.invalid", "project_id": "a" * 32, "root": str(tmp_path),
              "agent": "worker-device", "local_agent": "worker"}
    monkeypatch.setattr(cli, "Workspace", lambda: object())
    monkeypatch.setattr(cli, "_local_token", lambda args: "local-token")
    def denied(ws, token, *args):
        raise Denied("local identity revoked")
    monkeypatch.setattr(automatic_connection.device_agent, "owned_project_session", denied)
    monkeypatch.setattr(automatic_connection, "attach", lambda *a, **kw: pytest.fail("reenrollment"))
    with pytest.raises(Denied, match="revoked"):
        cli._context(SimpleNamespace(agent="worker", token_file=""), choice)


@pytest.mark.parametrize("agent,token_file", [("other", ""), ("worker", "explicit-project-token")])
def test_saved_alias_preserves_explicit_other_identity_and_token(monkeypatch, tmp_path, agent, token_file):
    choice = {"host": "https://board.invalid", "project_id": "a" * 32, "root": str(tmp_path),
              "agent": "worker-device", "local_agent": "worker"}
    monkeypatch.setattr(automatic_connection.device_agent, "owned_project_session", lambda *a: pytest.fail("alias substitution"))
    used = []
    remote = SimpleNamespace(token=lambda **kw: used.append(kw) or "project-token")
    monkeypatch.setattr(project_connection, "RemoteWorkspace", lambda *a, **kw: remote)
    monkeypatch.setattr(project_connection, "CanonicalWorkspace", lambda *a: object())
    cli._context(SimpleNamespace(agent=agent, token_file=token_file), choice)
    assert used == [{"agent": agent, "token_file": token_file}]


def _project_checkouts(repository, tmp_path):
    from ml_stack.net import git
    clone = tmp_path / "second-checkout"
    clone.mkdir()
    git.run(["init"], cwd=clone)
    git.run(["-c", "user.name=Test", "-c", "user.email=test@example.invalid",
             "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "fixture"], cwd=clone)
    for checkout in (repository, clone):
        git.run(["remote", "add", "origin", "https://forge.invalid/team/project.git"], cwd=checkout)
    return repository, clone


def _dev_device(state, number, checkout, udp, *, default_profile=False):
    from ml_stack.fleet import discovery, tls
    from ml_stack.fleet.api import Daemon, make_handler
    from ml_stack.fleet.framing import LimitedServer
    from ml_stack.fleet.jobs import JobRunner
    from ml_stack.fleet.onboard.joining import Joining
    from ml_stack.fleet.projects import ProjectRegistry
    from ml_stack.http import Server
    from ml_stack.workspace.remote_host import WorkspaceHost

    keyfile = state / "cluster.key"
    member = discovery.mint_cluster("development", keyfile, selection="automatic")
    files = state / "files"
    files.mkdir(parents=True)
    runner = JobRunner(state / "jobs", files)
    daemon = Daemon(runner, files, discovery.derive_token(member.key),
                    **({} if default_profile else {"cluster_key_path": keyfile}))
    daemon.tokens = lambda: {discovery.derive_token(row.key) for row in discovery.memberships(keyfile)}
    registry = ProjectRegistry(state / "fleet", f"node-{number}")
    daemon.projects, daemon.workspaces = registry, WorkspaceHost(registry)
    cert = tls.identity(state / "tls", f"node-{number}")
    daemon.joining = Joining(lambda: discovery.memberships(keyfile),
                             fingerprint=lambda: hashlib.sha256(cert.der).hexdigest(), log=lambda message: None)
    context = tls.server_context(cert)
    secured = LimitedServer((discovery.primary_ip(), 0), make_handler(daemon), tls=context)
    loopback = LimitedServer(("127.0.0.1", secured.server_port), make_handler(daemon), tls=context)
    local = Server(("127.0.0.1", 0), make_handler(daemon))
    registry.host = f"https://{discovery.primary_ip()}:{secured.server_port}"
    servers = [secured, loopback, local]
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    beacon = discovery.Beacon(name=f"node-{number}", machine=f"node-{number}",
                              port=secured.server_port, cert=cert.beacon)
    advertiser = discovery.Advertiser(beacon, member.key, cluster=member.group,
                                      port=udp, interval_s=30).start()
    return SimpleNamespace(state=state, checkout=checkout, keyfile=keyfile, daemon=daemon,
                           local_port=local.server_port, advertiser=advertiser, servers=servers, runner=runner)


def _start_convergence(device):
    from ml_stack.fleet import automatic_clusters, discovery
    stop = threading.Event()
    def refresh():
        device.advertiser.key = discovery.memberships(device.keyfile)[0].key
    thread = threading.Thread(target=automatic_clusters.converge,
                              args=(stop, refresh, device.keyfile), kwargs={"interval_s": 0.1}, daemon=True)
    thread.start()
    return stop, thread


@pytest.fixture
def dev_pair(repository, tmp_path, monkeypatch):
    from ml_stack import http
    from ml_stack.fleet import automatic_clusters, discovery
    from ml_stack.fleet.projects import identity
    from ml_stack.fleet.remote import Peer

    original_destinations = discovery._destinations
    monkeypatch.setattr(discovery, "_destinations", lambda group, port: [
        (address, interface) for address, interface in original_destinations(group, port)
        if address[0] != discovery.LOOPBACK and interface != discovery.LOOPBACK])
    checkouts = _project_checkouts(repository, tmp_path)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        udp = probe.getsockname()[1]
    original_discover, original_offers = Peer.discover, automatic_clusters.offers
    monkeypatch.setattr(Peer, "discover", lambda **kw: original_discover(
        **{**kw, "port": udp, "timeout_s": 0.3}))
    monkeypatch.setattr(automatic_clusters, "offers", lambda port=None: original_offers(udp))
    devices, convergence = [], []
    http._PINNED.clear()
    try:
        for number, checkout in enumerate(checkouts):
            devices.append(_dev_device(tmp_path / f"device-{number}", number, checkout, udp))
        original_keys = [discovery.memberships(device.keyfile)[0].key for device in devices]
        assert original_keys[0] != original_keys[1]
        convergence = [_start_convergence(device) for device in devices]
        yield devices, identity(repository)
    finally:
        for stop, thread in convergence:
            stop.set()
            thread.join(5)
        for device in devices:
            device.advertiser.stop()
            for server in device.servers:
                server.shutdown()
                server.server_close()
            device.runner.shutdown()
        http._PINNED.clear()


@pytest.mark.slow
def test_two_device_states_share_board_over_real_udp_and_tls_without_codes(dev_pair, monkeypatch):
    from ml_stack.fleet import discovery
    devices, project_id = dev_pair
    actors = []
    for device in devices:
        monkeypatch.setenv("ML_STACK_HOME", str(device.state / "client"))
        monkeypatch.setenv("ML_STACK_CLUSTER_KEY", str(device.keyfile))
        monkeypatch.setattr(automatic_connection, "HTTP_PORT", device.local_port)
        monkeypatch.chdir(device.checkout)
        args = SimpleNamespace(agent="worker", token_file="", model="", harness="")
        board, token = cli._context(args)
        actors.append((board, token, args.agent))
        assert project_connection.selected(device.checkout)["local_agent"] == "worker"
        again, repeated = cli._context(SimpleNamespace(agent="worker", token_file=""))
        assert repeated == token and again.remote.host == board.remote.host
    assert actors[0][0].remote.host == actors[1][0].remote.host
    assert discovery.memberships(devices[0].keyfile)[0].key == discovery.memberships(devices[1].keyfile)[0].key
    first, token, first_id = actors[0]
    second, other_token, other_id = actors[1]
    assert first_id != other_id
    sent = first.send(token, other_id, "note", "same Dev Board without a paste code")
    received = next(row for row in second.inbox(other_token) if row["seq"] == sent["seq"])
    assert received["from"] == first_id and received["to"] == other_id
    assert "same Dev Board without a paste code" in received["text"]
    assert received["authority"] == "none" and received["trust"] == "agent-claimed"
    authority = next(device.daemon.workspaces for device in devices
                     if urlsplit(device.daemon.projects.host).port == urlsplit(first.remote.host).port)
    ws = authority.workspace(project_id)
    assert all(ws.registry.info(actor)["model"] == "" for actor in (first_id, other_id))
    assert all(ws.registry.info(actor)["harness"] == "" for actor in (first_id, other_id))


@pytest.mark.parametrize("mode", ["dev", "prod"])
def test_default_profile_catalogue_and_automatic_board_use_real_udp_and_tls(repository, tmp_path, monkeypatch, mode):
    from ml_stack import http
    from ml_stack.fleet import automatic_clusters, discovery
    from ml_stack.fleet.projects import identity
    from ml_stack.fleet.remote import Peer
    original_destinations = discovery._destinations
    monkeypatch.setattr(discovery, "_destinations", lambda group, port: [
        (address, interface) for address, interface in original_destinations(group, port)
        if address[0] != discovery.LOOPBACK and interface != discovery.LOOPBACK])
    state = tmp_path / "default-device"
    monkeypatch.setenv("ML_STACK_HOME", str(state))
    monkeypatch.delenv("ML_STACK_CLUSTER_KEY", raising=False)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        udp = probe.getsockname()[1]
    original_discover = Peer.discover
    original_offers = automatic_clusters.offers
    monkeypatch.setattr(Peer, "discover", lambda **kw: original_discover(
        **{**kw, "port": udp, "timeout_s": 0.3}))
    monkeypatch.setattr(automatic_clusters, "offers", lambda port=None: original_offers(udp))
    device = _dev_device(state, 0, repository, udp, default_profile=True)
    if mode == "prod":
        device.advertiser.key = discovery.mint_cluster("development", mode="prod").key
    http._PINNED.clear()
    try:
        assert device.daemon.cluster_key_path is None
        assert discovery.key_path() == device.keyfile
        monkeypatch.setattr(automatic_connection, "HTTP_PORT", device.local_port)
        project_id = identity(repository)
        if mode == "prod":
            device.daemon.projects.register(repository, project_id)
        found = automatic_connection.discover(repository)
        if mode == "prod":
            assert found is None
            member = discovery.memberships()[0]
            peers = Peer.discover(key=member.key, group=member.group)
            assert peers
            assert all(automatic_connection.catalogue(node)["boards"] == [] for node in peers)
            return
        assert found is not None
        assert found["project_id"] == project_id
        assert found["cluster_key"] == ""
        assert device.daemon.projects.get(project_id).root == str(repository)
        _exchange_default_board(repository, project_id)
    finally:
        device.advertiser.stop()
        for server in device.servers:
            server.shutdown()
            server.server_close()
        device.runner.shutdown()
        http._PINNED.clear()


def _exchange_default_board(repository, project_id):
    from ml_stack.workspace.remote import RemoteWorkspace

    first = automatic_connection.connect(repository, "first")
    second = automatic_connection.connect(repository, "second")
    remote = RemoteWorkspace(first["host"], project_id)
    first_token = remote.token(agent=first["agent"])
    second_token = remote.token(agent=second["agent"])
    sent = remote.call("send", first_token, second["agent"], "note", "default profile connects without a code")
    received = next(row for row in remote.call("inbox", second_token) if row["seq"] == sent["seq"])
    assert received["from"] == first["agent"]
    assert received["to"] == second["agent"]
    assert "default profile connects without a code" in received["text"]


def test_setup_saved_standard_agent_attaches_without_device_registration(tmp_path, monkeypatch):
    from workspace_kit import Kit, clean_env

    from ml_stack.workspace import onboard, project, tokens
    from ml_stack.workspace.identity import AGENT

    root = tmp_path / 'project'
    root.mkdir()
    kit = Kit(clean_env(monkeypatch, tmp_path))
    token = kit.ws.registry.mint(onboard.SETUP, 'worker', AGENT, 3600)
    kit.ws.registry.set_project(kit.ws.auth(kit.owner), 'worker', project.describe(str(root)))
    tokens.store(kit.base, 'worker', token)
    before = kit.ws.registry.path.read_bytes()
    choice = {'automatic': True, 'root': str(root), 'host': 'https://board.invalid',
              'project_id': 'a' * 32, 'agent': ''}
    monkeypatch.setattr(cli, 'Workspace', lambda: kit.ws)
    monkeypatch.setattr(cli, '_local_token', lambda args: token)
    calls = []
    def attach(path, name, selected, *, claim):
        calls.append((path, name))
        return {**selected, 'agent': 'worker-device'}
    monkeypatch.setattr(automatic_connection, 'attach', attach)
    remote = SimpleNamespace(token=lambda **kwargs: 'private-project-session')
    monkeypatch.setattr(project_connection, 'RemoteWorkspace', lambda *args, **kwargs: remote)
    monkeypatch.setattr(project_connection, 'CanonicalWorkspace', lambda remote, capability: capability)
    actor = SimpleNamespace(agent='worker', token_file='')
    assert cli._context(actor, choice)[1] == 'private-project-session'
    assert calls == [(root, 'worker')] and actor.agent == 'worker-device'
    assert kit.ws.registry.path.read_bytes() == before
    assert not (kit.base / 'device-accounts.db').exists()


def test_saved_canonical_git_grant_refuses_another_repository(repository, tmp_path, monkeypatch):
    import json

    from workspace_kit import Kit, clean_env

    from ml_stack.net import git
    from ml_stack.workspace import device_agent, onboard, project, tokens
    from ml_stack.workspace.identity import AGENT

    kit = Kit(clean_env(monkeypatch, tmp_path))
    token = kit.ws.registry.mint(onboard.SETUP, 'worker', AGENT, 3600)
    grant = project.authoritative(str(repository))
    kit.ws.registry.set_project(kit.ws.auth(kit.owner), 'worker', grant)
    tokens.store(kit.base, 'worker', token)
    with device_agent.owned_project_session(kit.ws, token, 'worker', repository) as actor:
        assert actor.id == 'worker'
    foreign = tmp_path / 'foreign-repository'
    foreign.mkdir()
    git.run(['init'], cwd=foreign)
    (foreign / 'other.txt').write_text('unrelated project')
    git.run(['add', '--', 'other.txt'], cwd=foreign)
    git.run(['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
             '-c', 'commit.gpgsign=false', 'commit', '-m', 'fixture'], cwd=foreign)
    with pytest.raises(Denied, match='not authorized'), device_agent.owned_project_session(
            kit.ws, token, 'worker', foreign):
        pytest.fail('another repository received canonical project authorization')
    (foreign / '.ml-stack-project.json').write_text(json.dumps({
        'kind': 'project-checkout', 'project_id': grant['key']}))
    with pytest.raises(Denied, match='not authorized'), device_agent.owned_project_session(
            kit.ws, token, 'worker', foreign):
        pytest.fail('forged checkout metadata received canonical project authorization')
    monkeypatch.setenv('GIT_DIR', str(repository / '.git'))
    monkeypatch.setenv('GIT_WORK_TREE', str(repository))
    monkeypatch.setenv('GIT_COMMON_DIR', str(repository / '.git'))
    with pytest.raises(Denied, match='not authorized'), device_agent.owned_project_session(
            kit.ws, token, 'worker', foreign):
        pytest.fail('inherited Git selectors received canonical project authorization')
