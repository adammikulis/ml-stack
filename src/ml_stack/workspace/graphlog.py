"""Ordered graph message projections with per-origin hash-chain evidence."""

from __future__ import annotations

import time
from uuid import uuid4

from ml_stack.workspace.board_graph import BoardGraph
from ml_stack.workspace.chain import GENESIS, ChainBroken, Verdict, _digest


def verified(events, anchor=""):
    chains = {}
    for number, event in enumerate(events, 1):
        row = event["row"]
        prev, seq = chains.get(event["origin"], (row["prev"], row["seq"] - 1))
        if row["prev"] != prev or row["seq"] != seq + 1 or row["hash"] != _digest(prev, row):
            return Verdict(False, number - 1, prev, number, "graph event hash does not follow")
        chains[event["origin"]] = row["hash"], row["seq"]
    head = events[-1]["row"]["hash"] if events else GENESIS
    if anchor and anchor not in {GENESIS, *(e["row"]["hash"] for e in events)}:
        return Verdict(False, len(events), head, 0, "anchor is not in the graph")
    return Verdict(True, len(events), head)


class GraphLog:
    def __init__(self, base, stream, clock=time.time):
        self.graph = BoardGraph(base, clock)
        self.stream, self.clock = stream, clock

    def _project(self, events):
        sequences = {e["id"]: e["projection"] for e in events}
        return [{**e["row"], "seq": e["projection"], "id": e["id"], "origin": e["origin"],
                 **({"thread": sequences[e["thread_id"]], "thread_id": e["thread_id"]}
                    if e["thread_id"] in sequences else {})}
                for e in events if not e["retired"]]

    def rows(self):
        with self.graph.opened() as graph:
            events = self.graph._events(graph, self.stream)
            verdict = verified(events)
            if not verdict.ok:
                raise ChainBroken(verdict.reason)
            return self._project(events)

    def after(self, seq):
        return [row for row in self.rows() if row["seq"] > seq]

    def verify(self, anchor=""):
        try:
            with self.graph.opened() as graph:
                events = self.graph._events(graph, self.stream)
                verdict = verified(events, anchor)
                return Verdict(verdict.ok, sum(not e["retired"] for e in events), verdict.head,
                               verdict.broken_at, verdict.reason)
        except ChainBroken as error:
            return Verdict(False, 0, GENESIS, 0, str(error))

    def head(self):
        return self.verify().head

    def append(self, body):
        with self.graph.opened() as graph, graph.transaction():
            events = self.graph._events(graph, self.stream)
            verdict = verified(events)
            if not verdict.ok:
                raise ChainBroken(verdict.reason)
            scope = graph.get_doc("board-scope")
            origin = scope["replica"]
            local = [e for e in events if e["origin"] == origin]
            prior = local[-1]["row"] if local else {"seq": 0, "hash": GENESIS}
            row = {**body, "v": 1, "seq": prior["seq"] + 1, "prev": prior["hash"],
                   "ts": round(self.clock(), 3)}
            row["hash"] = _digest(row["prev"], row)
            thread_id = next((e["id"] for e in events if e["projection"] == body.get("thread")), "")
            event = {"id": f"{scope['workspace']}:event:{uuid4().hex}", "origin": origin,
                     "stream": self.stream, "row": row, "thread_id": thread_id}
            self.graph._insert(graph, event)
            return self._project(self.graph._events(graph, self.stream))[-1]

    def prune_prefix(self, drop):
        with self.graph.opened() as graph, graph.transaction():
            events = self.graph._events(graph, self.stream)
            verdict = verified(events)
            if not verdict.ok:
                raise ChainBroken(verdict.reason)
            count = 0
            for event in events:
                if event["retired"]:
                    continue
                row = self._project(events)[[e["id"] for e in events if not e["retired"]].index(event["id"])]
                if not drop(row):
                    break
                graph.upsert_node({**event, "retired": True})
                count += 1
            return count
