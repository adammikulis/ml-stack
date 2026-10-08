"""Verified network connection details for a project Board."""

import base64
import hashlib
import ipaddress
from pathlib import Path
from urllib.parse import urlsplit

from ml_stack.fleet.projects import lan_host
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.remote import RemoteWorkspace


def details(remote, token: str) -> dict:
    """Return a reachable pinned network endpoint and its authenticated identities."""
    who = remote.call("whoami", token)
    if who.get("project", {}).get("key") != remote.project_id:
        raise Denied("connection identity does not belong to the selected project")
    if not remote.device_cert:
        raise Denied("connection details require a TLS-authenticated device identity")
    parts = urlsplit(remote.host)
    try:
        address = ipaddress.ip_address(parts.hostname)
    except ValueError as exc:
        raise Denied("connection details require a numeric LAN address") from exc
    host = remote.host
    if address.is_loopback:
        host = lan_host(parts.port)
        if not host:
            raise Denied("no reachable network address is available; local access remains available")
    elif address.is_unspecified or address.is_multicast or not address.is_private:
        raise Denied("connection details require a reachable LAN address")
    peer = RemoteWorkspace(host, remote.project_id, cluster=remote.cluster,
                           cluster_key=Path(remote.cluster_key) if remote.cluster_key else None)
    if peer.device_cert != remote.device_cert:
        raise Denied("network endpoint does not match the authenticated project authority")
    verified = peer.call("whoami", token)
    if verified != who:
        raise Denied("network endpoint returned another project agent identity")
    return {"host": host, "port": urlsplit(host).port, "project_id": remote.project_id,
            "cluster": remote.cluster,
            "agent": who["id"], "tls_fingerprint": hashlib.sha256(
                base64.b64decode(remote.device_cert)).hexdigest(),
            "state": "verified"}
