"""Ordered graph message projections with per-origin hash-chain evidence."""

from __future__ import annotations

import time
from uuid import uuid4

from ml_stack.workspace.board_evidence import audience, checkpoints, seal, verified
from ml_stack.workspace.board_graph import BoardGraph
from ml_stack.workspace.chain import GENESIS, ChainBroken, Verdict, _digest


class GraphLog:
    def __init__(self, base, stream, clock=time.time):
        self.graph = BoardGraph(base, clock)
        self.stream, self.clock = stream, clock

    def _project(self, events, retired=None):
        sequences = {key: value["projection"] for key, value in (retired or {}).items()}
        sequences.update({e["id"]: e["projection"] for e in events})
        return [
            {
                **event["row"],
                "seq": event["projection"],
                "id": event["id"],
                "origin": event["origin"],
                "reply_to": sequences.get(event["reply_id"], 0),
                "reply_id": event["reply_id"],
                "thread": sequences.get(event["thread_id"], 0),
                "thread_id": event["thread_id"],
            }
            for event in events
            if not event["retired"]
        ]

    def rows(self):
        with self.graph.opened() as graph:
            events = self.graph._events(graph, self.stream)
            verdict = verified(events, bases=checkpoints(graph, self.stream))
            if not verdict.ok:
                raise ChainBroken(verdict.reason)
            return self._project(events, graph.get_doc("retired-events:" + self.stream))

    def after(self, seq):
        return [row for row in self.rows() if row["seq"] > seq]

    def verify(self, anchor=""):
        try:
            with self.graph.opened() as graph:
                events = self.graph._events(graph, self.stream)
                verdict = verified(events, anchor, checkpoints(graph, self.stream))
                return Verdict(
                    verdict.ok,
                    sum(not e["retired"] for e in events),
                    verdict.head,
                    verdict.broken_at,
                    verdict.reason,
                )
        except ChainBroken as error:
            return Verdict(False, 0, GENESIS, 0, str(error))

    def head(self):
        return self.verify().head

    def append(self, body):
        with self.graph.opened() as graph, graph.transaction():
            events = self.graph._events(graph, self.stream)
            verdict = verified(events, bases=checkpoints(graph, self.stream))
            if not verdict.ok:
                raise ChainBroken(verdict.reason)
            scope = graph.get_doc("board-scope")
            origin = scope["replica"]
            local = [e for e in events if e["origin"] == origin]
            prior = (
                local[-1]["row"]
                if local
                else checkpoints(graph, self.stream).get(origin, {"seq": 0, "hash": GENESIS})
            )
            row = {
                **body,
                "v": 1,
                "seq": prior["seq"] + 1,
                "prev": prior["hash"],
                "ts": round(self.clock(), 3),
            }
            projections = {event["projection"]: event["id"] for event in events}
            projections.update(
                {
                    row["projection"]: key
                    for key, row in (graph.get_doc("retired-events:" + self.stream) or {}).items()
                }
            )
            reply_id = projections.get(body.get("reply_to"), "")
            thread_id = projections.get(body.get("thread"), "")
            event_id = f"{scope['workspace']}:event:{uuid4().hex}"
            row.update(event_id=event_id, origin=origin, thread_id=thread_id, reply_id=reply_id)
            row["hash"] = _digest(row["prev"], row)
            event = {
                "id": event_id,
                "origin": origin,
                "stream": self.stream,
                "row": row,
                "thread_id": thread_id,
                "reply_id": reply_id,
            }
            self.graph._insert(graph, event)
            return self._project(
                self.graph._events(graph, self.stream),
                graph.get_doc("retired-events:" + self.stream),
            )[-1]

    def prune_prefix(self, drop):
        with self.graph.opened() as graph, graph.transaction():
            events = self.graph._events(graph, self.stream)
            verdict = verified(events, bases=checkpoints(graph, self.stream))
            if not verdict.ok:
                raise ChainBroken(verdict.reason)
            projected = self._project(events, graph.get_doc("retired-events:" + self.stream))
            cut = 0
            while cut < len(projected) and drop(projected[cut]):
                cut += 1
            bases = checkpoints(graph, self.stream)
            retired = graph.get_doc("retired-events:" + self.stream) or {}
            for event in events[:cut]:
                retired[event["id"]] = {
                    "origin": event["origin"],
                    "seq": event["row"]["seq"],
                    "hash": event["row"]["hash"],
                    "projection": event["projection"],
                    "audience": audience(event["row"]),
                }
                bases[event["origin"]] = {"seq": event["row"]["seq"], "hash": event["row"]["hash"]}
                graph.drop([event["id"], event["id"] + ":message"], force=True)
            graph.put_doc("origin-bases:" + self.stream, bases)
            references = {
                event[field] for event in events[cut:] for field in ("thread_id", "reply_id")
            }
            graph.put_doc(
                "retired-events:" + self.stream,
                {key: value for key, value in retired.items() if key in references},
            )
            seal(graph, self.stream)
            return cut
