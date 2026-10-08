"""Journal exchange between paired devices: the server side of pull and push, and the client."""

from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from typing import Any

from ml_stack.http import ServerError
from ml_stack.workspace import coordinator_client, journal_merge as rules, mesh_fold
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.service import Workspace

__all__ = ["PREFIX", "Target", "answer", "exchange", "paired_targets", "route", "sync"]

PREFIX = "/workspace/v1/mesh/"
VERSION = 1
MAX_ROWS = 400
MAX_ROUNDS = 20
MAX_REQUEST = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Target:
    """A paired device to exchange journals with."""

    fingerprint: str
    peer: Any


def _vector(doc: Any) -> dict[str, int]:
    if type(doc) is not dict:
        raise ValueError("a vector is an object")
    out = {}
    for origin, entry in doc.items():
        if type(origin) is not str or type(entry) is not dict or type(entry.get("seq")) is not int:
            raise ValueError("a vector entry names an origin and a sequence number")
        out[origin] = entry["seq"]
    return out


def _rows(mesh, since: dict[str, int], budget: int = MAX_ROWS) -> dict[str, list[dict[str, Any]]]:
    """For each origin the trusted rows past ``since``, starting one row early so the receiver
    can compare it with its copy, at most ``budget`` rows in all."""
    out: dict[str, list[dict[str, Any]]] = {}
    for origin in mesh.journals.origins():
        trusted = mesh.journals.trusted(origin)
        have = since.get(origin, 0)
        if origin in mesh.journals.damaged() or have >= len(trusted) or budget <= 0:
            continue
        out[origin] = trusted[max(0, have - 1):][:budget]
        budget -= len(out[origin])
    return out


def _trusted_vector(mesh) -> dict[str, dict[str, Any]]:
    out = {}
    for origin in mesh.journals.origins():
        trusted = mesh.journals.trusted(origin)
        if trusted:
            out[origin] = {"seq": trusted[-1]["seq"], "hash": trusted[-1]["hash"]}
    return out


def _take(mesh, journals: Any) -> tuple[int, dict[str, str]]:
    """Store what ``journals`` carries; returns the row count and the origins refused."""
    if type(journals) is not dict:
        raise ValueError("journals are an object")
    stored, refused = 0, {}
    for origin, rows in journals.items():
        if type(origin) is not str or type(rows) is not list:
            raise ValueError("a journal is a list of rows")
        try:
            stored += len(mesh.journals.ingest(origin, rows))
        except (rules.Damaged, rules.Gap) as bad:
            refused[origin] = str(bad)
    return stored, refused


def answer(ws, path: str, document: Any, fingerprint: str = "") -> dict[str, Any]:
    """The reply to a pull or a push from a paired device."""
    mesh = ws.mesh
    if type(document) is not dict or document.get("version") != VERSION or type(document.get("origin")) is not str:
        raise ValueError("a mesh request names its version and origin")
    peer = document["origin"]
    mesh.note_peer(fingerprint, peer)
    if path == PREFIX + "pull":
        since = _vector(document["vector"])
        mesh.acknowledge(peer, {o: {"seq": s} for o, s in since.items()})
        mesh.seal()
        return {"version": VERSION, "origin": mesh.origin, "vector": _trusted_vector(mesh),
                "journals": _rows(mesh, since)}
    stored, refused = _take(mesh, document.get("journals"))
    mesh_fold.apply(ws)
    return {"version": VERSION, "origin": mesh.origin, "stored": stored, "refused": refused,
            "vector": _trusted_vector(mesh)}


def route(handler, body: bytes | None) -> bool:
    """Serve /workspace/v1/mesh/pull and /push on the authenticated peer server."""
    path = handler.path.split("?")[0]
    if path not in (PREFIX + "pull", PREFIX + "push"):
        return False
    secure = isinstance(handler.connection, ssl.SSLSocket) or handler.client_address[0] in ("127.0.0.1", "::1")
    device = getattr(handler, "_workspace_device", None)
    if handler.command != "POST" or not secure or device is None:
        handler._send(403, {"error": "journal exchange needs a paired device over an encrypted connection"})
        return True
    try:
        if len(body or b"") > MAX_REQUEST:
            raise ValueError("the request exceeds its bound")
        handler._send(200, answer(Workspace(), path, json.loads(body or b"{}"), device.fingerprint))
    except (ValueError, KeyError, TypeError):
        handler._send(400, {"error": "invalid journal exchange"})
    except (OSError, RuntimeError):
        handler._send(503, {"error": "journal storage is unavailable"})
    return True


def paired_targets() -> list[Target]:
    """The active paired devices this device can authenticate to."""
    out = []
    for row in coordinator_client._paired_rows():
        try:
            peer = coordinator_client._device_peer({"endpoint": row["url"], "cert": row["certificate"]})
        except (Denied, KeyError):
            continue
        out.append(Target(str(row["fingerprint"]), peer))
    return out


def exchange(ws, target: Target) -> dict[str, Any]:
    """Pull what ``target`` holds that this device lacks, push what it lacks, fold the result."""
    mesh = ws.mesh
    result: dict[str, Any] = {"pulled": 0, "pushed": 0, "refused": {}, "rewound": [], "signed": mesh.seal()}
    remote: dict[str, int] = {}
    for _ in range(MAX_ROUNDS):
        reply = target.peer._json("POST", PREFIX + "pull", {
            "version": VERSION, "origin": mesh.origin, "vector": mesh.journals.vector()})
        mesh.note_peer(target.fingerprint, reply["origin"])
        remote = _vector(reply["vector"])
        result["rewound"] = sorted({*result["rewound"], *mesh.acknowledge(reply["origin"], reply["vector"])})
        stored, refused = _take(mesh, reply["journals"])
        result["pulled"] += stored
        result["refused"].update(refused)
        if not stored:
            break
    unsent = _rows(mesh, remote)
    if unsent:
        reply = target.peer._json("POST", PREFIX + "push", {"version": VERSION, "origin": mesh.origin,
                                                            "journals": unsent})
        result["pushed"] = int(reply["stored"])
        result["refused"].update(reply["refused"])
        mesh.acknowledge(reply["origin"], reply["vector"])
    mesh_fold.apply(ws)
    return result


def sync(ws, targets: list[Target] | None = None) -> list[dict[str, Any]]:
    """Exchange journals with every paired device; an unreachable device is reported, never raised."""
    out = []
    for target in paired_targets() if targets is None else targets:
        try:
            out.append({"device": target.fingerprint, **exchange(ws, target)})
        except (ServerError, OSError, ValueError, KeyError) as error:
            out.append({"device": target.fingerprint, "error": str(error)})
    return out
