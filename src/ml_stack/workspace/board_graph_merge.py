"""Bounded same-authority graph message exchange without membership grants."""

from __future__ import annotations

import json
import math
import re

from ml_stack.workspace.board_evidence import audience, checkpoints, verified
from ml_stack.workspace.board_graph import MAX_BYTES, MAX_EVENTS, identifier
from ml_stack.workspace.chain import GENESIS

HEX32 = re.compile(r"[0-9a-f]{32}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def export(graph_store):
    with graph_store.opened() as graph:
        events = graph_store._events(graph, "bus")
        bases = checkpoints(graph, "bus")
        if not verified(events, bases=bases).ok:
            raise ValueError("damaged message evidence cannot be exported")
        boards = [
            {key: node[key] for key in ("name", "project", "open", "title")}
            for node in graph.nodes("board")
        ]
        payload = {
            "version": 1,
            "workspace": graph.get_doc("board-scope")["workspace"],
            "boards": boards,
            "bases": bases,
            "events": [
                {
                    key: node[key]
                    for key in ("id", "origin", "stream", "row", "thread_id", "reply_id")
                }
                for node in events
            ],
        }
        bounded(payload)
        return payload


def bounded(payload):
    try:
        size = len(json.dumps(payload, allow_nan=False).encode())
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("invalid board exchange") from error
    if not isinstance(payload, dict) or size > MAX_BYTES:
        raise ValueError("board exchange exceeds its byte limit")
    if (
        set(payload) != {"version", "workspace", "boards", "bases", "events"}
        or payload["version"] != 1
        or not isinstance(payload["events"], list)
        or not isinstance(payload["bases"], dict)
    ):
        raise ValueError("invalid board exchange")
    if len(payload["events"]) > MAX_EVENTS or len(payload["bases"]) > MAX_EVENTS:
        raise ValueError("board exchange exceeds its event limit")


def valid_event(event, workspace):
    required = {"id", "origin", "stream", "row", "thread_id", "reply_id"}
    if (
        not isinstance(event, dict)
        or set(event) != required
        or event["stream"] != "bus"
        or not isinstance(event["row"], dict)
        or not all(isinstance(event[key], str) for key in ("id", "origin", "thread_id", "reply_id"))
    ):
        raise ValueError("invalid immutable board message event")
    legacy = event["origin"] == "legacy:" + workspace
    prefix = workspace + (":bus:" if legacy else ":event:")
    suffix = event["id"].removeprefix(prefix)
    if (
        not event["id"].startswith(prefix)
        or not (HEX64 if legacy else HEX32).fullmatch(suffix)
        or (not legacy and not HEX32.fullmatch(event["origin"]))
    ):
        raise ValueError("invalid immutable board message identity")
    row = event["row"]
    if legacy and event["id"] != identifier(workspace, "bus", row.get("hash")):
        raise ValueError("invalid legacy immutable message identity")
    strings = ("prev", "hash", "from", "to", "body", "type", "subject", "role", "held")
    if (
        row.get("kind") != "msg"
        or type(row.get("v")) is not int
        or row["v"] != 1
        or type(row.get("seq")) is not int
        or row["seq"] < 1
        or not all(isinstance(row.get(key), str) for key in strings)
        or not HEX64.fullmatch(row["prev"])
        or not HEX64.fullmatch(row["hash"])
        or not row["from"]
        or not row["to"]
        or type(row.get("ts")) not in (int, float)
        or not math.isfinite(row["ts"])
    ):
        raise ValueError("invalid board message row")
    for key in ("reply_to", "thread"):
        if key in row and (type(row[key]) is not int or row[key] < 0):
            raise ValueError("invalid message reply sequence")
    if not isinstance(row.get("mentions"), list) or any(
        not isinstance(item, str) for item in row["mentions"]
    ):
        raise ValueError("invalid message mentions")
    if "expires" in row and (
        type(row["expires"]) not in (int, float)
        or not math.isfinite(row["expires"])
        or row["expires"] < 0
    ):
        raise ValueError("invalid message expiry")
    if not isinstance(row.get("flags"), list) or any(
        not isinstance(flag, str) for flag in row["flags"]
    ):
        raise ValueError("invalid message flags")
    if any(
        key in row and not isinstance(row[key], str) for key in ("label", "model", "model_state")
    ):
        raise ValueError("invalid message display metadata")
    if "file" in row:
        file = row["file"]
        if (
            not isinstance(file, dict)
            or not all(isinstance(file.get(key), str) for key in ("id", "name", "type"))
            or type(file.get("size")) is not int
            or file["size"] < 0
            or type(file.get("text")) is not bool
        ):
            raise ValueError("invalid message file metadata")
    if not legacy and any(
        row.get(key) != event[field]
        for key, field in (
            ("event_id", "id"),
            ("origin", "origin"),
            ("thread_id", "thread_id"),
            ("reply_id", "reply_id"),
        )
    ):
        raise ValueError("message hash does not bind its immutable identity")


def validate_references(incoming, known, retired, workspace):
    by_sequence = {(event["origin"], event["row"]["seq"]): event["id"] for event in known.values()}
    by_sequence.update({(event["origin"], event["seq"]): key for key, event in retired.items()})
    for event in incoming:
        row = event["row"]
        for field in ("thread_id", "reply_id"):
            target = event[field]
            if event["origin"] == "legacy:" + workspace:
                sequence = (
                    row.get("thread", 0)
                    if field == "thread_id"
                    else row.get("reply_to") or row.get("thread", 0)
                )
                expected = by_sequence.get((event["origin"], sequence), "") if sequence else ""
                if target != expected or (sequence and not expected):
                    raise ValueError(
                        "legacy reply identity disagrees with original sequence evidence"
                    )
            if target:
                target_audience = (
                    retired[target]["audience"]
                    if target in retired
                    else (audience(known[target]["row"]) if target in known else "")
                )
                if target_audience != audience(row):
                    raise ValueError("reply target is absent or addresses another audience")


def authorized_boards(boards, descriptors):
    if not isinstance(descriptors, list) or len(descriptors) > 1000:
        raise ValueError("invalid board descriptors")
    names = set()
    for descriptor in descriptors:
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != {"name", "project", "open", "title"}
            or not isinstance(descriptor["name"], str)
            or descriptor["name"] not in boards
            or descriptor["name"] in names
            or type(descriptor["open"]) is not bool
            or not all(isinstance(descriptor[key], str) for key in ("project", "title"))
        ):
            raise ValueError("create or authorize the destination board before combining messages")
        local = boards[descriptor["name"]]
        if any(descriptor[key] != local[key] for key in ("project", "open", "title")):
            raise ValueError("conflicting board identity or visibility")
        names.add(descriptor["name"])
    return names


def combine(graph_store, payload):
    bounded(payload)
    with graph_store.opened() as graph, graph.transaction():
        workspace = graph.get_doc("board-scope")["workspace"]
        if payload["workspace"] != workspace:
            raise ValueError("board exchange belongs to another workspace authority")
        existing = graph_store._events(graph, "bus")
        local_bases = checkpoints(graph, "bus")
        boards = {node["name"]: node for node in graph.nodes("board")}
        names = authorized_boards(boards, payload["boards"])
        for origin, base in payload["bases"].items():
            if (
                not isinstance(origin, str)
                or not isinstance(base, dict)
                or set(base) != {"seq", "hash"}
                or type(base["seq"]) is not int
                or base["seq"] < 0
                or not isinstance(base["hash"], str)
                or not HEX64.fullmatch(base["hash"])
            ):
                raise ValueError("invalid origin checkpoint")
            known = local_bases.get(origin, {"seq": 0, "hash": GENESIS})
            if base != known:
                raise ValueError("origin checkpoint is not authorized by local history")
        events = payload["events"]
        incoming = {}
        for event in events:
            valid_event(event, workspace)
            if event["id"] in incoming:
                raise ValueError("duplicate immutable event identity")
            if event["row"]["to"].startswith("#") and event["row"]["to"] not in names:
                raise ValueError("message addresses an unauthorized board")
            incoming[event["id"]] = event
        if not verified(events, bases=payload["bases"]).ok:
            raise ValueError("imported origin chain is damaged")
        known_events = {event["id"]: event for event in existing} | incoming
        retired = graph.get_doc("retired-events:bus") or {}
        validate_references(events, known_events, retired, workspace)
        count = 0
        seen = {event["id"] for event in existing} | set(retired)
        remaining = list(events)
        predecessor, last_origin = {}, {}
        for event in events:
            predecessor[event["id"]] = last_origin.get(event["origin"], "")
            last_origin[event["origin"]] = event["id"]
        while remaining:
            ready = [
                event
                for event in remaining
                if all(not event[key] or event[key] in seen for key in ("thread_id", "reply_id"))
                and (not predecessor[event["id"]] or predecessor[event["id"]] in seen)
            ]
            if not ready:
                raise ValueError("reply graph contains a cycle")
            for event in ready:
                count += graph_store._insert(graph, event)
                seen.add(event["id"])
                remaining.remove(event)
        if not verified(graph_store._events(graph, "bus"), bases=local_bases).ok:
            raise ValueError("combined origin history forks")
        return count
