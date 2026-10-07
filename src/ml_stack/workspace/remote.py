"""A client of one authoritative shared project board."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from ml_stack import home, http, sealing
from ml_stack.fleet import tls
from ml_stack.fleet.discovery import derive_token, load_cluster_key, memberships
from ml_stack.fleet.onboard.lan import require_local_url
from ml_stack.fleet.remote import Peer, device_address
from ml_stack.graph.store import GraphStore
from ml_stack.http import ServerError, open_stream
from ml_stack.workspace import coordinator_client, device_metadata, tokens
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import Denied, valid_id


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
        self.mode = rows[0].mode if rows else "prod"
        self.cluster_id = hashlib.sha256(key).hexdigest()
        self.fleet_token = derive_token(key)
        self.device_cert = ""
        if parts.scheme == "https":
            peers = Peer.discover(key=key, timeout_s=2)
            matched = [peer for peer in peers if device_address(peer, self.host)]
            if len(matched) != 1:
                raise Denied("the project host was not authenticated by cluster discovery; check its address and cluster")
            self.device_cert = matched[0].beacon.cert if matched[0].beacon else ""
            if self.device_cert:
                http.pin(parts.netloc, tls.pinned_context(self.device_cert))
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
                             token=self._transport(action, payload), timeout=30, guard=guard,
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
        self._prepare_storage()
        result = self._request("join", {"code": code, "name": name,
                                        "model": model, "harness": harness})
        name, token = str(result["id"]), str(result["token"])
        path = tokens.store(self.base, name, token)
        return {"id": name, "project_id": self.project_id, "host": self.host,
                "token_file": str(path), "state": "connected"}

    def enroll(self, name: str, *, model: str, harness: str) -> dict:
        """Save a newly issued private Dev project agent capability."""
        self._prepare_storage()
        result = self._request("enroll", {"name": name, "model": model,
                                          "harness": harness, "cluster": self.cluster, "cluster_id": self.cluster_id,
                                          "device": device_metadata.current()})
        name, token = str(result["id"]), str(result["token"])
        if result.get("project_id") != self.project_id:
            raise Denied("agent enrollment returned another project")
        path = tokens.store(self.base, name, token)
        return {"id": name, "project_id": self.project_id, "host": self.host,
                "token_file": str(path), "state": "connected"}

    def renew(self, agent: str) -> dict:
        """Renew the existing private automatic capability for the current Dev cluster."""
        if self.mode != "dev" or not self.device_cert:
            raise Denied("agent renewal requires its authenticated Dev project authority")
        self._prepare_storage()
        lock = self.base / "remote-sessions.lock"
        self._safe_storage(lock)
        with held(lock):
            token = tokens.load(self.base, agent)
            result = self._request("renew", {"agent_token": token, "cluster": self.cluster,
                                             "cluster_id": self.cluster_id, "device": device_metadata.current()})
            if (result.get("id") != agent or result.get("project_id") != self.project_id
                    or result.get("cluster_id") != self.cluster_id):
                raise Denied("agent renewal returned another identity or project cluster")
            return result

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
        """Atomically reserve project areas and branches for the selected worker."""
        return self.call("native.reserve", self.token(agent=name), resources, label=label)

    def native_release(self, name: str, kind: str, relativekey: str) -> dict:
        """Release the selected worker's project resource."""
        return self.call("native.release", self.token(agent=name), kind, relativekey)

    def token(self, *, agent: str = "", token_file: str = "") -> str:
        if token_file:
            path = Path(token_file).expanduser()
            self._safe_storage(path)
            if not path.resolve().is_relative_to(tokens.directory(self.base).resolve()) or path.name == tokens.OWNER_FILE:
                raise Denied("remote operations use only private project agent capability files")
            return tokens.read_file(path)
        name = agent or os.environ.get(tokens.AGENT_ENV, "")
        if not name:
            raise Denied("select the local agent identity with --agent NAME")
        if not valid_id(name):
            raise Denied("select a valid local agent identity")
        self._prepare_storage()
        lock = self.base / "remote-sessions.lock"
        self._safe_storage(lock)
        with held(lock):
            return self._session_token(name)

    def _prepare_storage(self) -> None:
        self._safe_storage(self.base)
        self.base.mkdir(parents=True, mode=0o700, exist_ok=True)
        tokens.prepare(self.base)
        self._safe_storage(self.base)

    def _safe_storage(self, path: Path) -> None:
        for candidate in (self.base, tokens.directory(self.base), path):
            why = tokens.problem(candidate)
            if why not in {"", "missing"}:
                raise Denied(f"project session storage {candidate}: {why}")

    def _session_token(self, name: str) -> str:
        path = self.base / "remote-sessions.db"
        session = {}
        why = tokens.problem(path)
        if why not in {"", "missing"} and not why.startswith("mode "):
            raise Denied(f"project session records {path}: {why}")
        if path.exists():
            with GraphStore(path) as graph:
                session = next((node["attrs"] for node in graph.nodes("remote-session")
                                if name in (node["attrs"].get("name"), node["attrs"].get("id"))), {})
        ident = str(session.get("id", name))
        if not valid_id(ident):
            raise Denied("project session record holds an invalid agent identity")
        try:
            saved = tokens.load(self.base, ident)
        except Denied:
            self._safe_storage(tokens.directory(self.base) / ident.replace("/", "~"))
            saved = ""
        if saved:
            try:
                self.call("whoami", saved)
                if "/" not in ident and self.mode == "dev":
                    result = self._request("renew", {"agent_token": saved, "cluster": self.cluster,
                                                    "cluster_id": self.cluster_id,
                                                    "device": device_metadata.current()})
                    if (result.get("id") != ident or result.get("project_id") != self.project_id
                            or result.get("cluster_id") != self.cluster_id):
                        raise Denied("agent renewal returned another identity or project cluster")
                return saved
            except Denied as error:
                if not isinstance(error.__cause__, ServerError) or error.__cause__.status != 403:
                    raise
                if not session:
                    raise
        if "/" in ident:
            raise Denied("delegated project identities need a live parent-authorized credential")
        result = self._request("ensure", {"name": str(session.get("name", name)),
                                         "model": "", "harness": "",
                                         "project": {"key": self.project_id}, "agent_token": saved,
                                         "device": device_metadata.current()})
        ident, token = str(result["id"]), str(result["token"])
        tokens.store(self.base, ident, token)
        with GraphStore(path) as graph:
            graph.upsert_node({"id": f"session:{name}", "kind": "remote-session", "label": ident,
                               "attrs": {"name": str(session.get("name", name)), "id": ident}})
        return token

    def _transport(self, action: str, payload: dict) -> str:
        if action == "ensure":
            return self._device_transport()
        if action == "board":
            self._safe_storage(self.base)
            credential = str(payload.get("agent_token", ""))
            if credential.startswith(tokens.PREFIX):
                name = credential[len(tokens.PREFIX):].rsplit(".", 1)[0].split("/", 1)[0]
                path = self.base / "remote-sessions.db"
                why = tokens.problem(path)
                if why not in {"", "missing"} and not why.startswith("mode "):
                    raise Denied(f"project session records {path}: {why}")
                if path.exists():
                    with GraphStore(path) as graph:
                        if any(node["attrs"].get("id") == name
                               for node in graph.nodes("remote-session")):
                            return self._device_transport()
        return self.fleet_token

    def _device_transport(self) -> str:
        return coordinator_client._device_peer(
            {"endpoint": self.host, "cert": self.device_cert}).token
