"""Automatic Dev project discovery and canonical Board attachment."""

import hashlib
import ipaddress
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

from ml_stack import person, sealing, worktreerules
from ml_stack.fleet import automatic_clusters
from ml_stack.fleet.discovery import derive_token, memberships, primary_ip
from ml_stack.fleet.launch import HTTP_PORT
from ml_stack.fleet.project_client import catalogue
from ml_stack.fleet.projects import identity
from ml_stack.fleet.remote import Peer
from ml_stack.http import ServerError, ServerUnreachable, open_stream
from ml_stack.workspace import device_agent, tokens
from ml_stack.workspace.harness_seat import Seat
from ml_stack.workspace.identity import AGENT, Denied
from ml_stack.workspace.project_connection import bind, selected
from ml_stack.workspace.remote import RemoteWorkspace

MAX_PEERS = 8
DISCOVERY_SECONDS = 10.0
ADMISSION_SECONDS = 25.0


def _device_address(peer, declared):
    """Whether a Board origin names its authenticated discovery endpoint."""
    if declared == peer.base_url:
        return True
    try:
        observed, named = urlsplit(peer.base_url), urlsplit(declared)
        if (named.scheme != "https" or observed.scheme != "https" or named.port != observed.port
                or named.username or named.password or named.path not in ("", "/")
                or named.query or named.fragment):
            return False
        return (ipaddress.ip_address(observed.hostname).is_loopback
                and named.hostname == primary_ip())
    except ValueError:
        return False


def settle(member, cluster_key=None, port=None):
    """Converge visible Dev devices before choosing a project authority."""
    deadline = time.monotonic() + ADMISSION_SECONDS
    while True:
        member = automatic_clusters.ensure(cluster_key, mode="dev", port=port)
        if member.selection == "manual":
            return member
        offered = automatic_clusters.offers(port)
        cluster_id = hashlib.sha256(member.key).hexdigest()
        if all(row["cluster_id"] == cluster_id for host, row in offered):
            return member
        if time.monotonic() >= deadline:
            raise Denied("nearby Dev devices have not converged; retry when their cluster discovery is ready")
        time.sleep(0.1)


def cli_connection(args, connection, local_token, workspace):
    """Resolve this command's authenticated local agent on its Dev Board."""
    if getattr(args, "token_file", ""):
        return connection
    if connection.get("automatic"):
        local = workspace()
        actor = device_agent.owned_local(local, local_token(args))
        metadata = local.registry.info(actor.id)
        claim = (getattr(args, "model", "") or metadata["model"],
                 getattr(args, "harness", "") or metadata["harness"])
        connection = attach(Path(connection["root"]), actor.id, connection, claim=claim)
        args.agent = connection["agent"]
    elif connection.get("local_agent"):
        requested = args.agent or os.environ.get(tokens.AGENT_ENV, "")
        if requested == connection["local_agent"]:
            actor = device_agent.owned_local(workspace(), local_token(args))
            if actor.id != requested:
                raise Denied("the local agent does not match this project's saved identity")
            args.agent = connection["agent"]
    return refresh(connection, args.agent or os.environ.get(tokens.AGENT_ENV, ""))


def refresh(connection: dict, actor: str) -> dict:
    """Renew a saved automatic Dev connection when its cluster key changes."""
    if (not connection.get("agent") or actor not in {connection["agent"], connection.get("local_agent")}
            or not (connection.get("cluster_id") or connection.get("local_agent"))):
        return connection
    key_path = Path(connection["cluster_key"]) if connection.get("cluster_key") else None
    rows = memberships(key_path)
    if not rows or rows[0].mode != "dev":
        return connection
    if connection.get("cluster_id") == hashlib.sha256(rows[0].key).hexdigest():
        return connection
    remote = RemoteWorkspace(connection["host"], connection["project_id"],
                             cluster_key=key_path, cluster=rows[0].group)
    return bind(remote, Path(connection["root"]), connection["agent"], rows[0].group,
                local_agent=connection.get("local_agent", ""))


def _boards(peer, document, project_id):
    rows = document.get("boards", [])
    if not isinstance(rows, list) or len(rows) > 1000:
        raise Denied("the project host returned an invalid Board catalogue")
    found = []
    for row in rows:
        if not isinstance(row, dict):
            raise Denied("the project host returned an invalid Board entry")
        for field, maximum in (("id", 32), ("name", 256), ("machine", 256),
                               ("authority_machine", 256), ("board_host", 2048)):
            value = row.get(field)
            if not isinstance(value, str) or len(value) > maximum:
                raise Denied("the project host returned an invalid Board entry")
        if row["machine"] != document["machine"]:
            raise Denied("the project host advertised another device's Board")
        if row["id"] == project_id:
            found.append((peer, row))
    if len(found) > 1:
        raise Denied("the project host advertised duplicate Board entries")
    return found



def _register(root, member, project_id):
    endpoint = f"http://127.0.0.1:{HTTP_PORT}/workspace/v1/local-project"
    def destination(url):
        if url != endpoint:
            raise Denied("local project registration cannot redirect")
        return url
    try:
        with open_stream(endpoint, method="POST", token=derive_token(member.key),
                         data=json.dumps({"root": str(root.resolve()), "project_id": project_id}).encode(),
                         headers={"Content-Type": "application/json", sealing.HEADER: "2"},
                         timeout=2, guard=destination) as response:
            if not response.headers.get(sealing.HEADER):
                raise Denied("local project registration was not authenticated and sealed")
            limit = 65536 + sealing.NONCE_BYTES + 16
            raw = response.read(limit + 1)
            if len(raw) > limit:
                raise Denied("local project registration exceeds its response limit")
            result = json.loads(response.sealed.open(response.status, response.headers, raw))
            if not isinstance(result, dict) or result.get("id") != project_id:
                raise Denied("local project registration returned another project")
    except ServerUnreachable:
        return
    except ServerError as error:
        raise Denied("local Fleet project registration failed") from error

def discover(root: Path, *, cluster_key=None, cluster="", port=None):
    """Select one authenticated canonical host for this local Git project."""
    project_id = identity(root.resolve())
    rows = memberships(cluster_key)
    members = rows[:1] if rows and getattr(rows[0], "mode", "prod") == "dev" else []
    if cluster:
        members = [member for member in members if member.group == cluster]
    if not members:
        return None
    if len(members) != 1:
        raise Denied("select this project's Dev cluster with --cluster NAME")
    member = settle(members[0], cluster_key, port)
    if cluster and member.group != cluster:
        raise Denied("the selected Dev cluster changed during project discovery")
    _register(root, member, project_id)
    deadline = time.monotonic() + DISCOVERY_SECONDS
    peers = Peer.discover(key=member.key, group=member.group, timeout=2,
                          cluster_key_path=cluster_key, timeout_s=2, port=port)
    if len(peers) > MAX_PEERS:
        raise Denied("project discovery exceeds the device limit")
    found = []
    for peer in peers:
        if urlsplit(peer.base_url).scheme != "https":
            continue
        if time.monotonic() >= deadline:
            raise Denied("project discovery exceeded its time limit")
        document = catalogue(peer, deadline=deadline)
        if not peer.beacon or document["machine"] != peer.beacon.machine:
            raise Denied("the project catalogue does not match its authenticated device")
        found.extend(_boards(peer, document, project_id))
    if not found:
        return None
    authorities = {row["authority_machine"] for _, row in found if row["authority_machine"]}
    if len(authorities) > 1:
        raise Denied("this project has competing canonical Board authorities")
    if authorities:
        authority = next(iter(authorities))
        hosts = [(peer, row) for peer, row in found if row["machine"] == authority]
        if len(hosts) != 1:
            raise Denied("the canonical Board authority is unavailable or ambiguous")
        peer, row = hosts[0]
        declared = {candidate["board_host"] for _, candidate in found
                    if candidate["authority_machine"] == authority and candidate["board_host"]}
        if len(declared) != 1 or not _device_address(peer, next(iter(declared))):
            raise Denied("the canonical Board address does not match its authenticated device")
    else:
        peer, row = min(found, key=lambda item: (item[1]["machine"], item[0].base_url))
        if sum(candidate["machine"] == row["machine"] for _, candidate in found) != 1:
            raise Denied("the selected Board device is ambiguous")
    return {"host": peer.base_url, "project_id": project_id,
            "authority_machine": row["machine"], "cluster": member.group,
            "cluster_key": str(cluster_key) if cluster_key else ""}


def local_project(cwd: Path | None = None):
    """Discover the canonical Dev Board for the checkout containing this directory."""
    checkout = worktreerules.checkouts(cwd or Path.cwd())
    if checkout is None:
        return None
    root = checkout[0]
    choice = discover(root)
    return {**choice, "root": str(root), "agent": "", "automatic": True} if choice else None


def connect(root: Path, name: str, *, claim=("", ""), cluster_key=None, cluster=""):
    """Attach this project to its authenticated Dev Board with an agent capability."""
    choice = discover(root, cluster_key=cluster_key, cluster=cluster)
    if choice is None:
        raise Denied("no authenticated Dev device advertises this project's Board")
    return attach(root, name, choice, claim=claim)


def attach(root: Path, name: str, choice: dict, *, claim=("", "")):
    """Attach an agent to the selected authenticated project authority."""
    prior = selected(root)
    if prior and (prior["host"], prior["project_id"]) != (choice["host"], choice["project_id"]):
        raise Denied("this project already names another canonical Board")
    remote = RemoteWorkspace(choice["host"], choice["project_id"],
                             cluster_key=Path(choice["cluster_key"]) if choice.get("cluster_key") else None,
                             cluster=choice["cluster"])
    if prior and prior.get("agent") and (not name or name in (prior["agent"], prior.get("local_agent"))):
        return bind(remote, root, prior["agent"], choice["cluster"],
                    local_agent=prior.get("local_agent", ""))
    if not name:
        raise Denied("select your own agent name with --name NAME")
    joined = remote.enroll(name, model=claim[0], harness=claim[1],
                           authority_machine=choice["authority_machine"])
    return bind(remote, root, joined["id"], choice["cluster"], local_agent=name)


def startup(root: Path, name: str, parent: str = "", *, claim=("", "")) -> Seat | None:
    """Acquire a canonical project seat once when a native harness starts."""
    if not worktreerules.checkouts(root):
        return None
    prior = selected(root)
    choice = discover(root)
    if choice is None and prior is None:
        return None
    if prior and prior["project_id"] != identity(root):
        raise Denied("this checkout does not match its canonical Board project")
    if prior and choice and (prior["host"], prior["project_id"]) != (choice["host"], choice["project_id"]):
        raise Denied("this project already names another canonical Board")
    configuration = prior or choice
    remote = RemoteWorkspace(configuration["host"], configuration["project_id"],
                             cluster=configuration.get("cluster", ""),
                             cluster_key=Path(configuration["cluster_key"])
                             if configuration.get("cluster_key") else None)
    agent_started = bool(person.marked())
    if agent_started and not prior:
        raise Denied("an agent-started native session requires its parent's canonical Board connection")
    if prior and prior.get("agent"):
        actor = parent if agent_started else prior["agent"]
        if agent_started and actor != prior["agent"]:
            raise Denied("the native session parent does not match this project's connected agent")
        who = remote.call("whoami", remote.token(agent=actor))
        if who.get("id") != actor or who.get("role") != AGENT:
            raise Denied("native sessions delegate from their authenticated project agent")
        made = remote.delegate(actor, name)
        return Seat(made["id"], minted=True, base=remote.base,
                    remote=remote, lifecycle_base=remote.base)
    if agent_started or choice is None:
        raise Denied("this native session has no authenticated project parent")
    made = remote.enroll(f"native-{name}"[:48], model=claim[0], harness=claim[1],
                         authority_machine=choice["authority_machine"])
    bind(remote, root, made["id"], choice["cluster"])
    child = remote.delegate(made["id"], name)
    return Seat(child["id"], minted=True, base=remote.base,
                remote=remote, lifecycle_base=remote.base)
