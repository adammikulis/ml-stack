"""Requests people make about their own entries, and the edits a model proposes for them.

Nothing here changes a graph. A request becomes a proposal, and a proposal waits for a
person: `ml_stack.graph.review.Queue` is what lands an accepted one in the store.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ml_stack.files import read_json, write_json

#: what a request that arrived as a sentence in the chat is called in the review list
CHAT_KIND = "Asked in the chat"

PROPOSING = ("Somebody has asked for their own information in a graph to change. Propose the "
             "changes that would answer them, using the tools, one call per change, each with "
             "the reason in a sentence. Propose nothing you were not asked for, and nothing "
             "about anybody else.")


def read_requests(path: Path) -> list[dict[str, Any]]:
    """Every request on disk, oldest first; a line with no words, or not JSON, is skipped."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if str(row.get("text") or "").strip():
            out.append(row)
    return out


def key(request: Mapping[str, Any]) -> str:
    """What makes this request the same one across runs."""
    return f"{request.get('at', '')}|{request.get('text', '')}"


def edge_id(edge: Mapping[str, Any]) -> str:
    """``source|rel|target``."""
    return f"{edge['source']}|{edge['rel']}|{edge['target']}"


def as_prompt(request: Mapping[str, Any]) -> str:
    """The request with whatever the person pointed at, in one piece of text."""
    named = ", ".join(t.get("label", "") for t in request.get("targets") or [] if t.get("label"))
    kind = request.get("kind") or ""
    who = request.get("claimedLabel") or ""
    parts = [f"[{kind}]" if kind else "", request.get("text", "")]
    if who:
        parts.append(f"Asked by {who}, about their own information.")
    if named:
        parts.append(f"They pointed at: {named}.")
    return " ".join(p for p in parts if p).strip()


def considered(request: Mapping[str, Any], graph: Mapping[str, Any], client: Any,
               *, instructions: str = PROPOSING) -> list[Any]:
    """What the model would do about one request, as `Change`s it cannot carry out.

    `plan_edits` reads a request as operations on ids; this asks the same question with
    tools, so the model can say "these two are the same person" or "this link should not
    be there" in the graph's own words. A client that cannot hold a conversation is offered
    nothing and proposes nothing.
    """
    from ml_stack.graph.propose import proposing

    if not callable(getattr(client, "chat", None)):
        return []
    tools, gather = proposing(graph)
    reply = client.chat([{"role": "system", "content": instructions},
                         {"role": "user", "content": as_prompt(request)}],
                        tools=tools, think=False)
    return gather(getattr(reply, "tool_calls", None) or [])


def propose(requests: Sequence[Mapping[str, Any]], graph: Mapping[str, Any], client: Any,
            *, done: Mapping[str, Any] | None = None, log: Callable[[str], Any] = lambda _: None,
            instructions: str = PROPOSING) -> dict[str, Any]:
    """Ask the model what each new request means, as edits against ids that exist.

    Returns ``done`` with one proposal per request not already in it, keyed by `key`:
    the request's fields, ``concerns`` (why it might not be theirs to change), ``edits``
    (checked against the graph, unapplied) and ``status: proposed``.
    """
    from ml_stack.entities.edits import plan_edits
    from ml_stack.graph.concerns import concerns

    out = dict(done or {})
    nodes = {n["id"]: n["label"] for n in graph["nodes"]}
    edges = {edge_id(e): e["rel"] for e in graph["edges"]}
    for request in requests:
        k = key(request)
        if k in out:
            continue
        edits = plan_edits(as_prompt(request), nodes=nodes, edges=edges, client=client)
        listed = [{"op": e.op, "target": e.target, "name": e.name,
                   "value": e.value, "reason": e.reason} for e in edits]
        for change in considered(request, graph, client, instructions=instructions):
            listed.append({"op": change.op, "target": change.target, "other": change.other,
                           "name": change.name, "value": change.value, "reason": change.reason,
                           "problems": change.problems, "proposed": True})
        out[k] = {
            "at": request.get("at", ""),
            "kind": request.get("kind", ""),
            "claimed": request.get("claimed", ""),
            "claimedLabel": request.get("claimedLabel", ""),
            "attested": bool(request.get("attested")),
            "concerns": concerns(dict(request), listed, dict(graph)),
            "text": request.get("text", ""),
            "targets": request.get("targets") or [],
            "edits": listed,
            "status": "proposed",
        }
        log(f"request {request.get('at', '')!r}: {len(edits)} proposed edit(s)")
    return out


def from_chat(text: str, graph: Mapping[str, Any], client: Any, *, at: str = "",
              claimed: str = "", claimed_label: str = "",
              kind: str = CHAT_KIND) -> tuple[str, dict[str, Any]]:
    """One sentence typed into the chat, as a request the review queue already holds.

    Not attested -- the chat cannot tell who is typing -- so it lands exactly as an
    unticked form request does, with whatever `concerns` raises, through the same `propose`.
    """
    request = {"at": at or time.strftime("%FT%TZ", time.gmtime()), "kind": kind,
               "claimed": claimed, "claimedLabel": claimed_label, "attested": False,
               "text": " ".join(str(text).split())[:2000], "targets": []}
    k = key(request)
    return k, propose([request], graph, client)[k]


def load_or_build(requests_path: Path, proposals_path: Path, graph: Mapping[str, Any],
                  *, model: Path | str, lease: Mapping[str, Any], n_predict: int,
                  log: Callable[[str], Any] = lambda _: None) -> dict[str, Any]:
    """Propose for anything on disk not proposed for yet, writing the proposals back.
    No pending request means no server is put up."""
    requests = read_requests(requests_path)
    done = read_json(proposals_path, {})
    pending = [r for r in requests if key(r) not in done]
    if not pending:
        log(f"requests: {len(requests)} on file, none new")
        return done

    from ml_stack.client import Client
    from ml_stack.serve import serve

    log(f"requests: {len(pending)} to read")
    with serve(str(model), **lease) as server:
        client = Client(server.base_url, slot=0, n_predict=n_predict, timeout=300)
        out = propose(pending, graph, client, done=done, log=log)
    write_json(proposals_path, out)
    return out
