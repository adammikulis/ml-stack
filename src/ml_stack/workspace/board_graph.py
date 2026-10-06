"""Transactional board state and immutable message evidence in GraphStore."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from ml_stack.files import read_json
from ml_stack.graph.store import GraphStore
from ml_stack.workspace.chain import ChainBroken, ChainLog, _digest, held

MEMORY = 128 * 1024 * 1024
MAX_EVENTS = 10000
MAX_BYTES = 16 * 1024 * 1024


def identifier(workspace, kind, value):
    return f"{workspace}:{kind}:" + hashlib.sha256(str(value).encode()).hexdigest()


class BoardGraph:
    def __init__(self, base: Path, clock=time.time):
        self.base, self.clock = base, clock
        self.path = base / "board.db"
        self.guard = base / "board-graph.lock"

    @contextmanager
    def opened(self):
        with held(self.guard), GraphStore(self.path, buffer_pool_size=MEMORY) as graph:
            if not graph.get_doc("board-scope"):
                with held(self.base / "coordination.lock"), GraphStore(
                        self.base / "coordination.db", buffer_pool_size=MEMORY) as coordination:
                    nodes = coordination.nodes("workspace")
                    if len(nodes) > 1:
                        raise ValueError("multiple canonical workspace identities")
                    workspace = nodes[0]["id"] if nodes else f"workspace:{uuid4().hex}"
                    if not nodes:
                        coordination.upsert_node({"id": workspace, "kind": "workspace"})
                graph.put_doc("board-scope", {"workspace": workspace, "replica": uuid4().hex})
                graph.upsert_node({"id": workspace, "kind": "workspace"})
            self._defaults(graph)
            self._migrate(graph)
            yield graph

    def _migrate(self, graph):
        scope = graph.get_doc("board-scope")
        for stream in ("bus", "boards"):
            key = "legacy:" + stream
            legacy = ChainLog(self.base / f"{stream}.jsonl", self.clock)
            verdict = legacy.verify()
            saved = graph.get_doc(key)
            if not verdict.ok:
                raise ChainBroken(f"legacy {stream} evidence is damaged")
            if saved is not None:
                if saved["head"] != verdict.head:
                    raise ChainBroken(f"legacy {stream} changed after graph migration")
                continue
            with graph.transaction():
                for row in legacy.rows():
                    event = {"id": identifier(scope["workspace"], stream, row["hash"]),
                             "origin": "legacy:" + scope["workspace"], "stream": stream,
                             "row": row, "thread_id": "", "legacy_projection": row["seq"]}
                    if stream == "bus" and row.get("thread"):
                        prior = self._events(graph, stream)
                        event["thread_id"] = next((n["id"] for n in prior
                                                    if n["projection"] == row["thread"]), "")
                    self._insert(graph, event)
                graph.put_doc(key, {"head": verdict.head})
        if not graph.get_doc("legacy:cursors"):
            with graph.transaction():
                for path in (self.base / "cursors").glob("*.json"):
                    data = read_json(path, {})
                    if path.name.endswith(".marks.json"):
                        who = path.name[:-11].replace("~", "/")
                        for key, seq in data.get("marks", {}).items():
                            self._cursor(graph, who, key, int(seq))
                    else:
                        self._cursor(graph, path.stem, "inbox", int(data.get("seq", 0)))
                graph.put_doc("legacy:cursors", {"migrated": True})
        self._defaults(graph)

    def _defaults(self, graph):
        workspace = graph.get_doc("board-scope")["workspace"]
        for name, title in (("#general", "everyone"), ("#announcements", "terse one-line progress")):
            ident = identifier(workspace, "board", name)
            if not graph.has(ident):
                graph.upsert_node({"id": ident, "kind": "board", "name": name, "by": "",
                                   "open": True, "project": "", "title": title, "created": 0.0})
                graph.upsert_edge({"source": workspace, "rel": "HAS_BOARD", "target": ident})

    def _events(self, graph, stream):
        return sorted((n for n in graph.nodes("board-event") if n["stream"] == stream),
                      key=lambda n: n["projection"])

    def _insert(self, graph, event):
        existing = next((n for n in graph.nodes("board-event") if n["id"] == event["id"]), None)
        if existing:
            if any(existing[k] != event[k] for k in ("origin", "stream", "row", "thread_id")):
                raise ValueError("immutable board event collision")
            return False
        prior = self._events(graph, event["stream"])
        made = {**event, "kind": "board-event", "projection": event.get("legacy_projection") or (prior[-1]["projection"] + 1 if prior else 1),
                "retired": False}
        graph.upsert_node(made)
        workspace = graph.get_doc("board-scope")["workspace"]
        graph.upsert_edge({"source": workspace, "rel": "HAS_EVENT", "target": made["id"]})
        if made["stream"] == "boards":
            self._board_event(graph, made)
        else:
            self._message(graph, made)
        return True

    def _person(self, graph, name):
        workspace = graph.get_doc("board-scope")["workspace"]
        ident = identifier(workspace, "identity", name)
        if not graph.has(ident):
            graph.upsert_node({"id": ident, "kind": "board-identity", "name": name})
        return ident

    def _message(self, graph, event):
        row = event["row"]
        graph.upsert_node({"id": event["id"] + ":message", "kind": "board-message",
                           "event_id": event["id"]})
        mid = event["id"] + ":message"
        graph.upsert_edge({"source": self._person(graph, row.get("from", "")), "rel": "SENT", "target": mid})
        target = row.get("to", "")
        workspace = graph.get_doc("board-scope")["workspace"]
        destination = identifier(workspace, "board", target) if target.startswith("#") else self._person(graph, target)
        if graph.has(destination):
            graph.upsert_edge({"source": mid, "rel": "ADDRESSED_TO", "target": destination})
        if event["thread_id"]:
            graph.upsert_edge({"source": mid, "rel": "REPLIES_TO", "target": event["thread_id"] + ":message"})

    def _board_event(self, graph, event):
        row = event["row"]
        workspace = graph.get_doc("board-scope")["workspace"]
        if row.get("kind") == "board":
            bid = identifier(workspace, "board", row["name"])
            if row["op"] == "create":
                if graph.has(bid):
                    raise ValueError("conflicting board creation")
                graph.upsert_node({"id": bid, "kind": "board", "name": row["name"], "by": row["by"],
                                   "open": bool(row["open"]), "project": row.get("project", ""),
                                   "title": row.get("title", ""), "created": row["ts"]})
                graph.upsert_edge({"source": workspace, "rel": "HAS_BOARD", "target": bid})
                if row.get("project"):
                    project = identifier(workspace, "project", row["project"])
                    graph.upsert_node({"id": project, "kind": "board-project", "key": row["project"]})
                    graph.upsert_edge({"source": bid, "rel": "FOR_PROJECT", "target": project})
                if row["by"]:
                    graph.upsert_edge({"source": self._person(graph, row["by"]), "rel": "MEMBER_OF", "target": bid})
            elif row["op"] == "join":
                graph.upsert_edge({"source": self._person(graph, row["who"]), "rel": "MEMBER_OF", "target": bid})
            elif row["op"] == "leave":
                graph.remove_edge(self._person(graph, row["who"]), "MEMBER_OF", bid)
        elif row.get("kind") == "sub":
            ident = identifier(workspace, "subscription", json.dumps([row["who"], row["stype"], row["target"]]))
            node = {"id": ident, "kind": "board-subscription", "who": row["who"], "stype": row["stype"],
                    "target": row["target"], "active": row["op"] == "set", "mode": row.get("mode", ""),
                    "since": int(row.get("since", 0))}
            graph.upsert_node(node)
            graph.upsert_edge({"source": self._person(graph, row["who"]), "rel": "SUBSCRIBES", "target": ident})

    def _cursor(self, graph, who, key, seq):
        workspace = graph.get_doc("board-scope")["workspace"]
        ident = identifier(workspace, "cursor", json.dumps([who, key]))
        existing = next((n for n in graph.nodes("board-cursor") if n["id"] == ident), None)
        value = max(int(existing["seq"]) if existing else 0, seq)
        graph.upsert_node({"id": ident, "kind": "board-cursor", "who": who, "key": key, "seq": value})
        graph.upsert_edge({"source": self._person(graph, who), "rel": "READ_CURSOR", "target": ident})
        return value

    def cursor(self, who, key, seq=None):
        with self.opened() as graph:
            if seq is not None:
                with graph.transaction():
                    return self._cursor(graph, who, key, seq)
            return next((n["seq"] for n in graph.nodes("board-cursor") if n["who"] == who and n["key"] == key), 0)

    def marks(self, who):
        with self.opened() as graph:
            return {n["key"]: n["seq"] for n in graph.nodes("board-cursor") if n["who"] == who and n["key"] != "inbox"}

    def state(self):
        with self.opened() as graph:
            identities = {n["id"]: n["name"] for n in graph.nodes("board-identity")}
            membership = graph.edges("MEMBER_OF")
            boards = {n["name"]: {k: n[k] for k in ("by", "open", "project", "title", "created")}
                      | {"members": {identities[e["source"]] for e in membership if e["target"] == n["id"]}}
                      for n in graph.nodes("board")}
            subs, sinces = {}, {}
            for n in graph.nodes("board-subscription"):
                if n["active"]:
                    key = n["stype"], n["target"]
                    subs.setdefault(n["who"], {})[key] = n["mode"]
                    sinces.setdefault(n["who"], {})[key] = n["since"]
            self._validate_state(graph, boards, subs, sinces)
            return boards, subs, sinces

    def _validate_state(self, graph, boards, subs, sinces):
        expected = {name: {"by": "", "open": True, "project": "", "title": title,
                           "members": set(), "created": 0.0}
                    for name, title in (("#general", "everyone"),
                                        ("#announcements", "terse one-line progress"))}
        subscriptions, starts, chains = {}, {}, {}
        for event in self._events(graph, "boards"):
            row = event["row"]
            prev, seq = chains.get(event["origin"], (row["prev"], row["seq"] - 1))
            if row["prev"] != prev or row["seq"] != seq + 1 or row["hash"] != _digest(prev, row):
                raise ChainBroken("board graph authority evidence is damaged")
            chains[event["origin"]] = row["hash"], row["seq"]
            if row.get("kind") == "board":
                if row["op"] == "create":
                    expected[row["name"]] = {key: row.get(key, "") for key in ("by", "open", "project", "title")}
                    expected[row["name"]].update(created=row["ts"], members={row["by"]} if row["by"] else set())
                elif row["name"] in expected:
                    members = expected[row["name"]]["members"]
                    if row["op"] == "join":
                        members.add(row["who"])
                    elif row["op"] == "leave":
                        members.discard(row["who"])
            elif row.get("kind") == "sub":
                key = row["stype"], row["target"]
                mine = subscriptions.setdefault(row["who"], {})
                since = starts.setdefault(row["who"], {})
                if row["op"] == "set":
                    mine[key], since[key] = row["mode"], int(row.get("since", 0))
                else:
                    mine.pop(key, None)
                    since.pop(key, None)
        actual_subs = {who: rows for who, rows in subs.items() if rows}
        expected_subs = {who: rows for who, rows in subscriptions.items() if rows}
        actual_starts = {who: rows for who, rows in sinces.items() if rows}
        expected_starts = {who: rows for who, rows in starts.items() if rows}
        if boards != expected or actual_subs != expected_subs or actual_starts != expected_starts:
            raise ChainBroken("board graph relationships disagree with authority evidence")
