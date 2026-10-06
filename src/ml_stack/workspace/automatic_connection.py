"""Automatic Dev project discovery and canonical Board attachment."""

from pathlib import Path
from urllib.parse import urlsplit

from ml_stack.fleet.discovery import memberships
from ml_stack.fleet.project_client import catalogue
from ml_stack.fleet.projects import identity
from ml_stack.fleet.remote import Peer
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.project_connection import bind, selected
from ml_stack.workspace.remote import RemoteWorkspace


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
    members = [member for member in memberships(cluster_key)
               if getattr(member, "mode", "prod") == "dev"
               and (not cluster or member.group == cluster)]
    if not members:
        return None
    if len(members) != 1:
        raise Denied("select this project's Dev cluster with --cluster NAME")
    member = members[0]
    peers = Peer.discover(key=member.key, group=member.group,
                          cluster_key_path=cluster_key, timeout_s=2, port=port)
    found = []
    for peer in peers:
        if urlsplit(peer.base_url).scheme != "https":
            continue
        document = catalogue(peer)
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
