"""Origin checkpoints and integrity evidence for retained Board events."""

from __future__ import annotations

import hashlib
import json

from ml_stack.workspace.chain import GENESIS, ChainBroken, Verdict, _digest


class EvidenceBroken(ChainBroken):
    def __init__(self, reason: str, broken_at: int) -> None:
        super().__init__(reason)
        self.broken_at = broken_at


def envelope(event):
    return hashlib.sha256(
        json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def audience(row):
    scope = (
        ["board", row["to"]]
        if row["to"].startswith("#")
        else ["dm", *sorted({row["from"], row["to"]})]
    )
    return envelope(scope)


def events(graph, stream):
    return sorted(
        (node for node in graph.nodes("board-event") if node["stream"] == stream),
        key=lambda node: node["projection"],
    )


def checkpoints(graph, stream):
    return graph.get_doc("origin-bases:" + stream) or {}


def verified(rows, anchor="", bases=None):
    chains = {origin: (base["hash"], base["seq"]) for origin, base in (bases or {}).items()}
    for number, event in enumerate(rows, 1):
        row = event["row"]
        prev, seq = chains.get(event["origin"], (GENESIS, 0))
        if (
            row["prev"] != prev
            or row["seq"] != seq + 1
            or row["hash"] != _digest(prev, row)
            or any(
                row.get(key) != event[field]
                for key, field in (
                    ("event_id", "id"),
                    ("origin", "origin"),
                    ("thread_id", "thread_id"),
                    ("reply_id", "reply_id"),
                )
                if "event_id" in row
            )
        ):
            return Verdict(False, number - 1, prev, number, "graph event hash does not follow")
        chains[event["origin"]] = row["hash"], row["seq"]
    head = rows[-1]["row"]["hash"] if rows else GENESIS
    if anchor and anchor not in {GENESIS, *(event["row"]["hash"] for event in rows)}:
        return Verdict(False, len(rows), head, 0, "anchor is not in the graph")
    return Verdict(True, len(rows), head)


def seal(graph, stream):
    rows = events(graph, stream)
    graph.put_doc(
        "event-evidence:" + stream,
        {
            "events": {event["id"]: envelope(event) for event in rows},
            "retired": graph.get_doc("retired-events:" + stream) or {},
            "bases": checkpoints(graph, stream),
            "scope": graph.get_doc("board-scope"),
            "projection": graph.get_doc("projection:" + stream) or {"seq": 0},
        },
    )


def validate(graph, stream):
    rows = events(graph, stream)
    evidence = graph.get_doc("event-evidence:" + stream)
    if evidence != {
        "events": {event["id"]: envelope(event) for event in rows},
        "retired": graph.get_doc("retired-events:" + stream) or {},
        "bases": checkpoints(graph, stream),
        "scope": graph.get_doc("board-scope"),
        "projection": graph.get_doc("projection:" + stream) or {"seq": 0},
    }:
        recorded = evidence.get("events", {}) if isinstance(evidence, dict) else {}
        changed = next(
            (
                number
                for number, event in enumerate(rows, 1)
                if recorded.get(event["id"]) != envelope(event)
            ),
            0,
        )
        raise EvidenceBroken(
            "board graph event envelopes disagree with integrity evidence", changed
        )
    if not verified(rows, bases=checkpoints(graph, stream)).ok:
        raise ChainBroken("board graph origin history is damaged")
