"""Authenticated project discovery and source checkout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml_stack import sealing
from ml_stack.http import Sealed, ServerError, open_stream

from .project_source import MAX_ARCHIVE, ProjectError, checkout, project_id
from .remote import Peer


def read(peer: Peer, path: str, limit: int) -> bytes:
    """Read one bounded signed and sealed project response."""
    with open_stream(peer.base_url + path, token=peer.token,
                     headers={sealing.HEADER: "2"}, timeout=30) as response:
        if not response.headers.get(sealing.HEADER):
            raise ProjectError("Project response was not authenticated and sealed")
        data = response.read(limit + 1024)
        if len(data) >= limit + 1024:
            raise ProjectError("Project response exceeds its size limit")
        opened = getattr(response, "sealed", Sealed()).open(response.status, response.headers, data)
        if len(opened) > limit:
            raise ProjectError("Project response exceeds its size limit")
        return opened


def catalogue(peer: Peer) -> dict:
    result = json.loads(read(peer, "/workspace/v1/projects", 1 << 20))
    if not isinstance(result, dict) or not isinstance(result.get("projects"), list):
        raise ProjectError("Peer returned an invalid project catalogue")
    for project in result["projects"]:
        project_id(project["id"])
        if project.get("authority_machine") != result.get("machine"):
            raise ProjectError("Peer advertised another device's project authority")
    return result


def available(ui) -> dict:
    projects, devices = [], []
    if ui.projects:
        projects = [{**p, "is_self": True, "peer": "", "state": "available"}
                    for p in ui.projects.list()]
    for peer in Peer.discover(cluster_key_path=ui.cluster_key_path, timeout_s=1, port=ui.discovery_port):
        if peer.beacon and ui.projects and peer.beacon.machine == ui.projects.machine:
            continue
        try:
            found = catalogue(peer)
            projects.extend({**p, "is_self": False, "peer": peer.base_url,
                             "host": peer.base_url, "state": "available"} for p in found["projects"])
            devices.append({"name": peer.name, "peer": peer.base_url, "state": "available"})
        except (ServerError, ProjectError, ValueError, OSError) as exc:
            devices.append({"name": peer.name, "peer": peer.base_url,
                            "state": "upgrade_required" if getattr(exc, "status", 0) in {404, 501} else "offline",
                            "error": str(exc)})
    for project in projects:
        authorities = {p["authority_machine"] for p in projects if p["id"] == project["id"]}
        if len(authorities) > 1:
            project["state"] = "conflict"
    return {"projects": projects, "devices": devices,
            "candidates": ui.projects.candidates() if ui.projects else [],
            "local": ui.projects.list() if ui.projects else []}


def receive(peer: Peer, identifier: str, base: Path) -> Path:
    rows = [p for p in catalogue(peer)["projects"] if p["id"] == project_id(identifier)]
    if len(rows) != 1:
        raise ProjectError("Project is not published by this device")
    project = rows[0]
    data = read(peer, f"/workspace/v1/projects/{identifier}/snapshot?hash={project['source_hash']}", MAX_ARCHIVE)
    if hashlib.sha256(data).hexdigest() != project["archive_sha256"]:
        raise ProjectError("Project archive digest mismatch")
    return checkout(data, identifier, project["source_hash"], base,
                    authority={"machine": project["authority_machine"], "host": peer.base_url})


def route(request) -> bool:
    ui, path, method = request.ui, request.path, request.method
    if not path.startswith("/ui/projects") or not ui.projects:
        return False
    suffix = path.removeprefix("/ui/projects").strip("/").split("/")
    try:
        if path == "/ui/projects" and method == "GET":
            request.send(200, available(ui))
        elif path == "/ui/projects" and method == "POST":
            body = request.body()
            if body.get("action") != "share":
                raise ProjectError("Choose share explicitly")
            request.send(200, {"project": ui.projects.share(body.get("candidate_id", ""), body.get("name", "")).public()})
        elif len(suffix) == 1 and method == "PATCH":
            if request.body().get("shared") is not False:
                raise ProjectError("Only explicit unsharing is accepted")
            ui.projects.unshare(suffix[0])
            request.send(200, {"ok": True})
        elif len(suffix) == 2 and suffix[1] == "checkout" and method == "POST":
            selected = request.body().get("peer")
            peers = [p for p in Peer.discover(cluster_key_path=ui.cluster_key_path, timeout_s=1,
                                             port=ui.discovery_port) if p.base_url == selected]
            if len(peers) != 1:
                raise ProjectError("Choose a currently authenticated cluster device")
            found = available(ui)
            if any(p["id"] == suffix[0] and p["state"] == "conflict" for p in found["projects"]):
                raise ProjectError("Project has conflicting workspace authorities")
            result = receive(peers[0], suffix[0], ui.root / "project-checkouts")
            request.send(200, {"project_id": suffix[0], "checkout": str(result),
                               "board_host": selected, "attached": False})
        else:
            return False
    except (ProjectError, ServerError, ValueError, OSError) as exc:
        request.send(400, {"error": str(exc)})
    return True
