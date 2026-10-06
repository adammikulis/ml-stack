"""Bounded same-authority graph message exchange without membership grants."""

from __future__ import annotations

import json

from ml_stack.workspace.board_graph import MAX_BYTES, MAX_EVENTS
from ml_stack.workspace.graphlog import verified


def export(graph_store):
    with graph_store.opened() as graph:
        events = graph_store._events(graph, "bus")
        if not verified(events).ok:
            raise ValueError("damaged message evidence cannot be exported")
        boards = [{key: node[key] for key in ("name", "project", "open", "title")}
                  for node in graph.nodes("board")]
        payload = {"version": 1, "workspace": graph.get_doc("board-scope")["workspace"],
                   "boards": boards, "events": [
                       {key: node[key] for key in ("id", "origin", "stream", "row", "thread_id")}
                       for node in events]}
        bounded(payload)
        return payload


def bounded(payload):
    if not isinstance(payload, dict) or len(json.dumps(payload).encode()) > MAX_BYTES:
        raise ValueError("board exchange exceeds its byte limit")
    if payload.get("version") != 1 or not isinstance(payload.get("events"), list):
        raise ValueError("invalid board exchange")
    if len(payload["events"]) > MAX_EVENTS:
        raise ValueError("board exchange exceeds its event limit")


def combine(graph_store, payload):
    bounded(payload)
    with graph_store.opened() as graph, graph.transaction():
        workspace = graph.get_doc("board-scope")["workspace"]
        if payload.get("workspace") != workspace:
            raise ValueError("board exchange belongs to another workspace authority")
        existing = graph_store._events(graph, "bus")
        if not verified(existing).ok:
            raise ValueError("local message evidence is damaged")
        boards = {node["name"]: node for node in graph.nodes("board")}
        if not isinstance(payload.get("boards"), list) or len(payload["boards"]) > 1000:
            raise ValueError("invalid board descriptors")
        for descriptor in payload["boards"]:
            if not isinstance(descriptor, dict) or descriptor.get("name") not in boards:
                raise ValueError("create or authorize the destination board before combining messages")
            local = boards[descriptor["name"]]
            if any(descriptor.get(key) != local[key] for key in ("project", "open", "title")):
                raise ValueError("conflicting board identity or visibility")
        events = payload["events"]
        seen = {event["id"] for event in existing}
        incoming = set()
        required = {"id", "origin", "stream", "row", "thread_id"}
        for event in events:
            if (not isinstance(event, dict) or set(event) != required
                    or event["stream"] != "bus" or not isinstance(event["row"], dict)
                    or not isinstance(event["id"], str) or not event["id"].startswith(workspace + ":")
                    or not isinstance(event["origin"], str) or len(event["origin"]) > 256
                    or not isinstance(event["thread_id"], str) or event["id"] in incoming):
                raise ValueError("invalid immutable board message event")
            row = event["row"]
            if (row.get("kind") != "msg" or type(row.get("seq")) is not int
                    or not all(isinstance(row.get(key), str) for key in ("prev", "hash", "from", "to"))):
                raise ValueError("invalid board message row")
            if row["to"].startswith("#") and row["to"] not in boards:
                raise ValueError("message addresses an unauthorized board")
            incoming.add(event["id"])
        if not verified(events).ok:
            raise ValueError("imported origin chain is damaged")
        for event in events:
            if event["thread_id"] and event["thread_id"] not in seen | incoming:
                raise ValueError("reply target is absent from the exchanged graph")
        count = 0
        remaining = list(events)
        predecessor = {}
        last_origin = {}
        for event in events:
            predecessor[event["id"]] = last_origin.get(event["origin"], "")
            last_origin[event["origin"]] = event["id"]
        while remaining:
            ready = [event for event in remaining
                     if (not event["thread_id"] or event["thread_id"] in seen)
                     and (not predecessor[event["id"]] or predecessor[event["id"]] in seen)]
            if not ready:
                raise ValueError("reply graph contains a cycle")
            for event in ready:
                count += graph_store._insert(graph, event)
                seen.add(event["id"])
                remaining.remove(event)
        if not verified(graph_store._events(graph, "bus")).ok:
            raise ValueError("combined origin history forks")
        return count
