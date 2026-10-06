"""A client of one authoritative shared project board."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from ml_stack import home, sealing
from ml_stack.fleet.discovery import derive_token, load_cluster_key, memberships
from ml_stack.fleet.onboard.lan import require_local_url
from ml_stack.fleet.remote import Peer
from ml_stack.http import ServerError, open_stream
from ml_stack.workspace import tokens
from ml_stack.workspace.identity import Denied


class RemoteWorkspace:
    """Keep a private agent capability for a selected project host."""

    def __init__(self, host: str, project_id: str, *, cluster_key: Path | None = None,
                 cluster: str = "") -> None:
        parts = urlsplit(host)
        if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username
                or parts.password or parts.query or parts.fragment or parts.path not in {"", "/"}):
            raise ValueError("a project host is an http(s) LAN address without a path")
        if not re.fullmatch("[a-f0-9]{32}", project_id):
            raise ValueError("invalid shared project ID")
        require_local_url(host)
        self.host, self.project_id = host.rstrip("/"), project_id
        self.cluster_key = str(cluster_key) if cluster_key else ""
        self.endpoint = f"{self.host}/workspace/v1/projects/{project_id}"
        rows = memberships(cluster_key)
        if cluster:
            rows = [row for row in rows if row.group == cluster]
            if len(rows) != 1:
                raise Denied("select a cluster this device has joined")
        elif len(rows) > 1:
            raise Denied("select this project's cluster with --cluster NAME")
        key = rows[0].key if rows else load_cluster_key(cluster_key)
        if key is None:
            raise Denied("join the host's cluster before attaching its project board")
        self.cluster = rows[0].group if rows else cluster
        self.cluster_id = hashlib.sha256(key).hexdigest()
        self.fleet_token = derive_token(key)
        if parts.scheme == "https":
            peers = Peer.discover(key=key, timeout_s=2)
            if not any(peer.base_url.rstrip("/") == self.host for peer in peers):
                raise Denied("the project host was not authenticated by cluster discovery; check its address and cluster")
        label = hashlib.sha256(f"{self.host}/{project_id}".encode()).hexdigest()
        self.base = home.state("workspace-remote", label)

    def _request(self, action: str, payload: dict) -> dict:
        def guard(url: str) -> str:
            if url != f"{self.endpoint}/{action}":
                raise Denied("a project board cannot redirect agent capabilities")
            require_local_url(url)
            return url
        url = guard(f"{self.endpoint}/{action}")
        try:
            with open_stream(url, data=json.dumps(payload).encode(), method="POST",
                             token=self.fleet_token, timeout=30, guard=guard,
                             headers={"Content-Type": "application/json", sealing.HEADER: "2"}) as response:
                if not response.headers.get(sealing.HEADER):
                    raise Denied("project board response was not authenticated and sealed")
                limit = 512 * 1024
                wire_limit = limit + sealing.NONCE_BYTES + 16
                raw = response.read(wire_limit + 1)
                if len(raw) > wire_limit:
                    raise ValueError("project board response exceeds the size limit")
                opened = response.sealed.open(response.status, response.headers, raw)
                if len(opened) > limit:
                    raise ValueError("project board response exceeds the size limit")
                result = json.loads(opened)
                if not isinstance(result, dict):
                    raise ValueError("project board returned no object")
                return result
        except ServerError as exc:
            if exc.status in {404, 501}:
                raise Denied("this host has no shared project board endpoint; upgrade or select its board host") from exc
            raise Denied(f"project board unavailable: {exc}") from exc

    def join(self, code: str, name: str, *, model: str = "", harness: str = "") -> dict:
        result = self._request("join", {"code": code, "name": name,
                                        "model": model, "harness": harness})
        name, token = str(result["id"]), str(result["token"])
        path = tokens.store(self.base, name, token)
        return {"id": name, "project_id": self.project_id, "host": self.host,
                "token_file": str(path), "state": "connected"}

    def enroll(self, name: str, *, model: str, harness: str, authority_machine: str) -> dict:
        """Save a newly issued private Dev project agent capability."""
        result = self._request("enroll", {"name": name, "model": model,
                                          "harness": harness, "cluster": self.cluster, "cluster_id": self.cluster_id,
                                          "authority_machine": authority_machine})
        name, token = str(result["id"]), str(result["token"])
        if result.get("project_id") != self.project_id:
            raise Denied("agent enrollment returned another project")
        path = tokens.store(self.base, name, token)
        return {"id": name, "project_id": self.project_id, "host": self.host,
                "token_file": str(path), "state": "connected"}

    def call(self, operation: str, token: str, *args, **kwargs):
        return self._request("board", {"agent_token": token, "operation": operation,
                                       "args": list(args), "kwargs": kwargs, "cluster": self.cluster})["result"]

    def delegate(self, parent: str, name: str) -> dict:
        """Store a private bounded child capability delegated by the selected parent."""
        result = self.call("delegate", self.token(agent=parent), name)
        child = f"{parent}/{name}"
        if result.get("id") != child or result.get("project_id") != self.project_id:
            raise Denied("delegation returned another project or identity")
        path = tokens.store(self.base, child, str(result["token"]))
        return {"id": child, "token_file": str(path), "expires": result["expires"]}

    def self_revoke(self, name: str) -> dict:
        """Revoke the selected agent's own capability and remove its private token file."""
        result = self.call("revoke_self", self.token(agent=name))
        if result.get("id") != name or result.get("revoked") is not True:
            raise Denied("self revocation returned another identity")
        (tokens.directory(self.base) / name.replace("/", "~")).unlink()
        return result

    def native_reserve(self, name: str, resources: list, label: str = "") -> list:
        """Atomically reserve canonical project areas and branches for the selected worker."""
        return self.call("native.reserve", self.token(agent=name), resources, label=label)

    def native_release(self, name: str, kind: str, relativekey: str) -> dict:
        """Release the selected worker's canonical project resource."""
        return self.call("native.release", self.token(agent=name), kind, relativekey)

    def token(self, *, agent: str = "", token_file: str = "") -> str:
        if token_file:
            path = Path(token_file).expanduser().resolve()
            if not path.is_relative_to(tokens.directory(self.base).resolve()) or path.name == tokens.OWNER_FILE:
                raise Denied("remote operations use only private project agent capability files")
            return tokens.read_file(path)
        name = agent or os.environ.get(tokens.AGENT_ENV, "")
        if not name:
            raise Denied("attach a project agent and select it with --agent NAME")
        return tokens.load(self.base, name)
