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

    def join_device(self, installation: str = "") -> dict:
        """Join this project workspace using the authenticated Fleet device membership."""
        installation = installation or home.machine_id()
        name = f"device-{installation}"
        existing = tokens.load(self.base, name) if (tokens.directory(self.base) / name).exists() else ""
        payload = {"installation": installation}
        if existing:
            payload["token"] = existing
        result = self._request("device", payload)
        name, token = str(result.get("id") or ""), str(result.get("token") or "")
        if (result.get("project_id") != self.project_id or name != f"device-{installation}"
                or not token):
            raise Denied("the project coordinator returned an invalid device enrollment")
        path = tokens.store(self.base, name, token)
        return {"id": name, "project_id": self.project_id, "host": self.host,
                "token_file": str(path), "state": "connected"}

    def call(self, operation: str, token: str, *args, **kwargs):
        return self._request("board", {"agent_token": token, "operation": operation,
                                       "args": list(args), "kwargs": kwargs})["result"]

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
