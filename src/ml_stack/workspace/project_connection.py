"""Private project connections and canonical workspace dispatch."""

import json
from pathlib import Path

from ml_stack import home
from ml_stack.files import read_json, write_json
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import AGENT, Denied, Identity
from ml_stack.workspace.remote import RemoteWorkspace
from ml_stack.workspace.remote_host import METHODS


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
        raise Denied("canonical board connection record is unreadable; local fallback is disabled") from exc


def bind(remote: RemoteWorkspace, root: Path, agent: str, cluster: str = "") -> dict:
    """Bind a local project root to an authenticated canonical board identity."""
    who = remote.call("whoami", remote.token(agent=agent))
    if who.get("role") != AGENT or who.get("project", {}).get("key") != remote.project_id:
        raise Denied("connect using this project's scoped agent identity")
    agent = who["id"]
    path = home.state("workspace-connections.json")
    root = root.resolve()
    metadata = read_json(root / ".ml-stack-project.json", {})
    authority = metadata.get("authority", {})
    if metadata.get("kind") == "project-checkout" and (
            metadata.get("project_id") != remote.project_id
            or (authority.get("host") and authority["host"] != remote.host)):
        raise Denied("this checkout already names another canonical board")
    made = {"host": remote.host, "project_id": remote.project_id, "agent": agent,
            "cluster": cluster, "cluster_key": remote.cluster_key}
    with held(path.with_suffix(".lock")):
        connections = _saved()
        existing = connections.get(str(root))
        if existing and (existing["host"], existing["project_id"]) != (made["host"], made["project_id"]):
            raise Denied("this project already uses another canonical board")
        connections[str(root)] = made
        write_json(path, connections)
        path.chmod(0o600)
    return {"root": str(root), **made, "state": "connected"}


def selected(cwd: Path | None = None) -> dict | None:
    """Return the closest project connection or verified checkout authority."""
    current = (cwd or Path.cwd()).resolve()
    connections = _saved()
    for root in (current, *current.parents):
        configured = connections.get(str(root))
        metadata_path = root / ".ml-stack-project.json"
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
                return configured
            if authority.get("host"):
                return {"host": authority["host"], "project_id": metadata["project_id"],
                        "agent": "", "cluster": authority.get("cluster", "")}
            raise Denied("this shared checkout has no canonical board connection; attach it before using workspace commands")
        if configured:
            return configured
    return None


class Operations:
    def __init__(self, parent, prefix: str = ""):
        self.parent, self.prefix = parent, prefix

    def __getattr__(self, name):
        operation = self.prefix + name
        if operation not in METHODS:
            raise Denied(f"{operation} is unavailable on the canonical board; local fallback is disabled")
        return lambda token, *args, **kwargs: self.parent.remote.call(operation, token, *args, **kwargs)


class CanonicalWorkspace(Operations):
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

    def info(self, name):
        who = self.remote.call("whoami", self.token)
        if name != who["id"]:
            raise Denied("only your own capability metadata is available")
        return who

    def model_of(self, name, label=""):
        info = self.info(name)
        return info.get("model", ""), info.get("model_state", "")

    def registered(self):
        return self.remote.call("agents", self.token)

    def listing(self, owner="", kind=""):
        rows = self.remote.call("claims", self.token)
        return [row for row in rows if (not owner or row["owner"] == owner)
                and (not kind or row["kind"] == kind)]

    def who_owns(self, kind, key):
        return self.remote.call("who", self.token, kind, key)
