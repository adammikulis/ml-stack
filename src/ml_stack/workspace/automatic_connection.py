"""Automatic Dev project discovery and canonical Board attachment."""

import time
from pathlib import Path
from urllib.parse import urlsplit

from ml_stack import person, worktreerules
from ml_stack.fleet.discovery import memberships
from ml_stack.fleet.project_client import catalogue
from ml_stack.fleet.projects import identity
from ml_stack.fleet.remote import Peer
from ml_stack.workspace.harness_seat import Seat
from ml_stack.workspace.identity import AGENT, Denied
from ml_stack.workspace.project_connection import bind, selected
from ml_stack.workspace.remote import RemoteWorkspace

MAX_PEERS = 8
DISCOVERY_SECONDS = 10.0


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
    member = members[0]
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
        if len(declared) != 1 or next(iter(declared)) != peer.base_url:
            raise Denied("the canonical Board address does not match its authenticated device")
    else:
        peer, row = min(found, key=lambda item: (item[1]["machine"], item[0].base_url))
        if sum(candidate["machine"] == row["machine"] for _, candidate in found) != 1:
            raise Denied("the selected Board device is ambiguous")
    return {"host": peer.base_url, "project_id": project_id,
            "authority_machine": row["machine"], "cluster": member.group,
            "cluster_key": str(cluster_key) if cluster_key else ""}


def connect(root: Path, name: str, *, model="", harness="", cluster_key=None,
            cluster="", port=None):
    """Attach this project to its authenticated Dev Board with an agent capability."""
    choice = discover(root, cluster_key=cluster_key, cluster=cluster, port=port)
    if choice is None:
        raise Denied("no authenticated Dev device advertises this project's Board")
    prior = selected(root)
    if prior and (prior["host"], prior["project_id"]) != (choice["host"], choice["project_id"]):
        raise Denied("this project already names another canonical Board")
    remote = RemoteWorkspace(choice["host"], choice["project_id"],
                             cluster_key=cluster_key, cluster=choice["cluster"])
    if prior and prior.get("agent") and (not name or prior["agent"] == name):
        return bind(remote, root, prior["agent"], choice["cluster"])
    if not name:
        raise Denied("select your own agent name with --name NAME")
    joined = remote.enroll(name, model=model, harness=harness,
                           authority_machine=choice["authority_machine"])
    return bind(remote, root, joined["id"], choice["cluster"])


def startup(root: Path, name: str, parent: str = "") -> Seat | None:
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
    made = remote.enroll(f"native-{name}"[:48], model="", harness="",
                         authority_machine=choice["authority_machine"])
    bind(remote, root, made["id"], choice["cluster"])
    child = remote.delegate(made["id"], name)
    return Seat(child["id"], minted=True, base=remote.base,
                remote=remote, lifecycle_base=remote.base)
