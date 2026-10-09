"""Private project connections and workspace dispatch."""

import json
import os
from pathlib import Path

from poolhouse import home, worktreerules
from poolhouse.files import read_json, writing
from poolhouse.fleet import discovery, project_client, projects, remote
from poolhouse.fleet.remote import same_machine_host
from poolhouse.http import ServerError
from poolhouse.net import git
from poolhouse.windows_private import restrict
from poolhouse.workspace import project_session
from poolhouse.workspace.chain import held
from poolhouse.workspace.identity import AGENT, Denied, Identity
from poolhouse.workspace.remote import RemoteWorkspace
from poolhouse.workspace.remote_protocol import METHODS


def _saved() -> dict:
    path = home.state("workspace-connections.json")
    if not path.exists():
        return {}
    try:
        connections = json.loads(path.read_text())
        if not isinstance(connections, dict) or any(
                not isinstance(value, dict) or not all(isinstance(value.get(key), str) and value[key]
                for key in ("host", "project_id", "agent"))
                for value in connections.values()):
            raise ValueError("invalid connections")
        return connections
    except (OSError, ValueError) as exc:
        raise Denied("board connection record is unreadable; local fallback is disabled") from exc


def _save(path: Path, connections: dict, before: str) -> None:
    """Write the connections record, unless it is the text just read."""
    text = json.dumps(connections)
    if text == before:
        return
    with writing(path) as tmp:
        restrict(tmp) if os.name == "nt" else tmp.chmod(0o600)
        tmp.write_text(text, encoding="utf-8")


def bind(remote: RemoteWorkspace, root: Path, agent: str, cluster: str = "", **options: str) -> dict:
    """Bind a local project root to an authenticated board identity."""
    if set(options) - {"local_agent", "agent_token"}:
        raise TypeError("unsupported project binding option")
    local_agent, agent_token = options.get("local_agent", ""), options.get("agent_token", "")
    prior = selected(root)
    if (prior and prior.get("agent") == agent and prior["host"] == remote.host
            and prior["project_id"] == remote.project_id and getattr(remote, "mode", "prod") == "dev"
            and (prior.get("cluster_id") or prior.get("local_agent"))
            and prior.get("cluster_id") != remote.cluster_id):
        remote.renew(agent)
    who = remote.call("whoami", agent_token or remote.token(agent=agent))
    if agent_token and who.get("id") != agent:
        raise Denied("the saved project capability belongs to another agent")
    if who.get("role") != AGENT or who.get("project", {}).get("key") != remote.project_id:
        raise Denied("connect using this project's scoped agent identity")
    agent = who["id"]
    path = home.state("workspace-connections.json")
    root = root.resolve()
    metadata = read_json(root / ".poolhouse-project.json", {})
    authority = metadata.get("authority", {})
    if metadata.get("kind") == "project-checkout" and (
            metadata.get("project_id") != remote.project_id
            or (authority.get("host") and authority["host"] != remote.host
                and not same_machine_host(authority["host"], remote.host))):
        raise Denied("this checkout already names another board")
    made = {"host": remote.host, "project_id": remote.project_id, "agent": agent,
            "cluster": cluster, "cluster_key": remote.cluster_key}
    if who.get("project", {}).get("cluster_id"):
        made["cluster_id"] = who["project"]["cluster_id"]
    if local_agent:
        made["local_agent"] = local_agent
    session = project_session.current(local_agent) if local_agent else ""
    expected = project_session.name(local_agent) if session else ""
    if session and agent != expected and not agent.startswith(expected + "-"):
        session = ""
    if session:
        made["session"] = session
    with held(path.with_suffix(".lock")):
        connections = _saved()
        before = json.dumps(connections)
        existing = connections.get(str(root))
        if existing and existing["host"] != made["host"] and same_machine_host(existing["host"], made["host"]):
            made["host"] = existing["host"]  # the recorded name for this machine stays the name
        if existing and (existing["host"], existing["project_id"]) != (made["host"], made["project_id"]):
            raise Denied("this project already uses another board")
        if session:
            existing = existing or made.copy()
            sessions = existing.setdefault("sessions", {})
            if session in sessions and sessions[session].get("agent") != agent:
                raise Denied("this native session already has another shared project identity")
            sessions[session] = made
            connections[str(root)] = existing
        else:
            if existing and existing.get("sessions"):
                made["sessions"] = existing["sessions"]
            connections[str(root)] = made
        _save(path, connections, before)
    return {"root": str(root), **made, "state": "connected"}


def selected(cwd: Path | None = None, *, local_agent: str = "") -> dict | None:
    """Return the closest project connection or verified checkout authority."""
    current = (cwd or Path.cwd()).resolve()
    connections = _saved()
    for root in (current, *current.parents):
        configured = connections.get(str(root))
        if configured and not (local_agent and local_agent != "codex"
                               and configured.get("local_agent") == local_agent):
            configured = project_session.connection(configured)
        metadata_path = root / ".poolhouse-project.json"
        metadata = read_json(metadata_path, {})
        if not isinstance(metadata, dict):
            raise Denied("shared checkout metadata is invalid; local fallback is disabled")
        if metadata_path.exists() and metadata.get("kind") != "project-checkout":
            raise Denied("shared checkout metadata is invalid; local fallback is disabled")
        if metadata.get("kind") == "project-checkout":
            authority = metadata.get("authority", {})
            if configured and (configured["project_id"] != metadata.get("project_id")
                               or (authority.get("host") and configured["host"] != authority["host"])):
                raise Denied("checkout and saved board connection disagree")
            if configured:
                return {**configured, "root": str(root)}
            if authority.get("host"):
                return {"host": authority["host"], "project_id": metadata["project_id"],
                        "agent": "", "cluster": authority.get("cluster", ""), "root": str(root)}
            raise Denied("this shared checkout has no board connection; attach it before using workspace commands")
        if configured:
            return {**configured, "root": str(root)}
    checkout = worktreerules.checkouts(current)
    if checkout and checkout[0] != checkout[1]:
        configured = connections.get(str(checkout[1]))
        if configured:
            if configured["project_id"] != projects.identity(checkout[0]):
                raise Denied("this worktree differs from its shared project Board")
            chosen = project_session.connection(configured)
            if chosen.get("session"):
                return {**chosen, "root": str(checkout[0])}
            return {**configured, "agent": "", "local_agent": "", "root": str(checkout[0]),
                    "automatic": True}
    return None


def _find_authority(project_id: str) -> tuple[str, str] | None:
    candidates = []
    for member in discovery.memberships():
        peers = remote.Peer.discover(key=member.key, group=member.group, timeout_s=1)
        published = []
        for peer in peers:
            try:
                rows = project_client.catalogue(peer)["projects"]
            except (OSError, ValueError, ServerError):
                continue
            published.extend(row for row in rows if row["id"] == project_id)
        boards = {row["board_host"] for row in published if row["board_host"]}
        if len(boards) > 1:
            raise Denied("this project has conflicting workspace authorities on the network")
        if not boards:
            continue
        board = next(iter(boards))
        hosts = [peer for peer in peers if remote.device_address(peer, board)]
        if len(hosts) != 1:
            raise Denied("the project's workspace authority is missing or ambiguous on this network")
        candidates.append((member.group, hosts[0].base_url))
    if not candidates:
        return None
    if len(set(candidates)) != 1:
        raise Denied("this project appears on more than one cluster; select one workspace authority")
    return candidates[0]


def auto_attach(cwd: Path | None = None) -> dict | None:
    """Discover this Git project's one workspace through an enrolled cluster."""
    current = (cwd or Path.cwd()).resolve()
    try:
        root = Path(git.run(["rev-parse", "--show-toplevel"], cwd=current).stdout.strip()).resolve()
        project_id = projects.identity(root)
    except (OSError, ValueError, git.GitFailed):
        return None
    authority = _find_authority(project_id)
    if authority is None:
        return None
    cluster, host = authority
    remote = RemoteWorkspace(host, project_id, cluster=cluster)
    return {"host": remote.host, "project_id": project_id, "agent": "",
            "cluster": cluster, "cluster_key": remote.cluster_key, "root": str(root)}


class Operations:
    def __init__(self, parent, prefix: str = ""):
        self.parent, self.prefix = parent, prefix

    def __getattr__(self, name):
        operation = self.prefix + name
        if operation not in METHODS:
            raise Denied(f"{operation} is unavailable on the board; local fallback is disabled")
        return lambda token, *args, **kwargs: self.parent.remote.call(operation, token, *args, **kwargs)


class BoardWorkspace(Operations):
    """The supported local CLI operations backed by one remote project board."""

    def __init__(self, remote: RemoteWorkspace, token: str):
        self.remote, self.token = remote, token
        super().__init__(self)
        self.board = Operations(self, "board.")
        self.registry = self
        self.claims = self

    def auth(self, token):
        who = self.remote.call("whoami", token)
        return Identity(who["id"], who["role"], who.get("parent", ""), tuple(who["can"]))

    def spawn(self, token, harness, session, model=""):
        return self.remote.spawn(self.auth(token).id, harness, session, model)

    def info(self, name):
        who = self.remote.call("whoami", self.token)
        if name != who["id"]:
            raise Denied("only your own capability metadata is available")
        return who

    def model_of(self, name):
        info = self.info(name)
        return info.get("model", ""), info.get("model_state", "")

    def status(self):
        registered = self.registered()
        return {"agents": [row["id"] for row in registered], "registered": registered,
                "claims": self.listing(), "project": self.remote.project_id, "state": "connected"}

    def registered(self):
        return self.remote.call("agents", self.token)

    def listing(self, owner="", kind=""):
        rows = self.remote.call("claims", self.token)
        return [row for row in rows if (not owner or row["owner"] == owner)
                and (not kind or row["kind"] == kind)]

    def who_owns(self, kind, key):
        return self.remote.call("who", self.token, kind, key)
