"""Journal exchange between paired devices: the server side of pull and push, and the client."""

from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from typing import Any

from poolhouse.http import ServerError
from poolhouse.workspace import coordinator_client, journal_merge as rules, mesh_fold
from poolhouse.workspace.identity import AGENT, Denied
from poolhouse.workspace.service import Workspace

__all__ = ["PREFIX", "Target", "answer", "exchange", "paired_targets", "route", "run", "sync"]

PREFIX = "/workspace/v1/mesh/"
VERSION = 1
MAX_ROWS = 400
MAX_ROUNDS = 20
MAX_REQUEST = 4 * 1024 * 1024
MAX_BATCH_BYTES = 2 * 1024 * 1024
MAX_ROW_BYTES = 128 * 1024
MAX_REQUEST_ORIGINS = 16


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
        if not rules.valid_origin(origin) or type(entry) is not dict or type(entry.get("seq")) is not int:
            raise ValueError("a vector entry names an origin and a sequence number")
        out[origin] = entry["seq"]
    return out


def _rows(mesh, since: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    """For each origin the trusted rows past ``since``, starting one row early so the receiver
    can compare it with its copy, within the row and byte limits of one request."""
    out: dict[str, list[dict[str, Any]]] = {}
    rows, size = 0, 0
    for origin in mesh.journals.origins():
        trusted = mesh.journals.trusted(origin)
        have = since.get(origin, 0)
        if origin in mesh.journals.damaged() or have >= len(trusted):
            continue
        for row in trusted[max(0, have - 1):]:
            size += len(json.dumps(row))
            if rows >= MAX_ROWS or (size > MAX_BATCH_BYTES and rows):
                return out
            out.setdefault(origin, []).append(row)
            rows += 1
    return out


def _trusted_vector(mesh) -> dict[str, dict[str, Any]]:
    out = {}
    for origin in mesh.journals.origins():
        trusted = mesh.journals.trusted(origin)
        if trusted:
            out[origin] = {"seq": trusted[-1]["seq"], "hash": trusted[-1]["hash"]}
    return out


def _take(mesh, journals: Any, fingerprint: str) -> tuple[int, dict[str, str]]:
    """Store what ``journals`` carries; returns the row count and the origins refused."""
    if type(journals) is not dict or len(journals) > MAX_REQUEST_ORIGINS:
        raise ValueError("journals are a small object")
    stored, refused, total = 0, {}, 0
    for origin, rows in journals.items():
        if not rules.valid_origin(origin) or type(rows) is not list:
            raise ValueError("a journal is a list of rows under an origin id")
        total += len(rows)
        if total > 2 * MAX_ROWS or any(len(json.dumps(r)) > MAX_ROW_BYTES for r in rows):
            raise ValueError("the request exceeds its bound")
        try:
            stored += len(mesh.journals.ingest(origin, rows, mesh.owns(fingerprint, origin)))
        except (rules.Damaged, rules.Gap, rules.Quota) as bad:
            refused[origin] = type(bad).__name__
    return stored, refused


def answer(ws, path: str, document: Any, fingerprint: str) -> dict[str, Any]:
    """The reply to a pull or a push from the paired device ``fingerprint``."""
    mesh = ws.mesh
    if type(document) is not dict or document.get("version") != VERSION or not rules.valid_origin(document.get("origin")):
        raise ValueError("a mesh request names its version and origin")
    if not mesh.bind_peer(fingerprint, document["origin"]):
        raise Denied("this device writes a different journal than the one it presented before")
    if path == PREFIX + "pull":
        since = _vector(document["vector"])
        mesh.acknowledge(fingerprint, document["vector"])
        mesh.seal()
        return {"version": VERSION, "origin": mesh.origin, "vector": _trusted_vector(mesh),
                "journals": _rows(mesh, since)}
    stored, refused = _take(mesh, document.get("journals"), fingerprint)
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
    except Denied:
        handler._send(403, {"error": "journal exchange refused"})
    except (ValueError, KeyError, TypeError, OverflowError):
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
    result: dict[str, Any] = {"pulled": 0, "pushed": 0, "refused": {}, "rewound": False, "signed": mesh.seal()}
    remote: dict[str, int] = {}
    for _ in range(MAX_ROUNDS):
        reply = target.peer._json("POST", PREFIX + "pull", {
            "version": VERSION, "origin": mesh.origin, "vector": mesh.journals.vector()})
        if not rules.valid_origin(reply.get("origin")) or not mesh.bind_peer(target.fingerprint, reply["origin"]):
            raise ValueError("the device presented a different journal than before")
        remote = _vector(reply["vector"])
        result["rewound"] = mesh.acknowledge(target.fingerprint, reply["vector"]) or result["rewound"]
        stored, refused = _take(mesh, reply["journals"], target.fingerprint)
        result["pulled"] += stored
        result["refused"].update(refused)
        if not stored:
            break
    unsent = _rows(mesh, remote)
    if unsent:
        reply = target.peer._json("POST", PREFIX + "push", {"version": VERSION, "origin": mesh.origin,
                                                            "journals": unsent})
        result["pushed"] = int(reply["stored"])
        result["refused"].update({o: str(r)[:40] for o, r in reply["refused"].items()})
        mesh.acknowledge(target.fingerprint, reply["vector"])
    mesh_fold.apply(ws)
    return result


def sync(ws, targets: list[Target] | None = None) -> list[dict[str, Any]]:
    """Exchange journals with every paired device; an unreachable device is reported, never raised."""
    out = []
    for target in paired_targets() if targets is None else targets:
        try:
            out.append({"device": target.fingerprint, **exchange(ws, target)})
        except (ServerError, OSError, ValueError, KeyError, TypeError):
            out.append({"device": target.fingerprint, "error": "unreachable or refused"})
    return out


def run(ws, token: str) -> list[dict[str, Any]]:
    """`sync` for the token's owner; an agent token cannot start signing or network calls."""
    if ws.auth(token).role == AGENT:
        raise Denied("a person or a lead exchanges journals")
    return sync(ws)
