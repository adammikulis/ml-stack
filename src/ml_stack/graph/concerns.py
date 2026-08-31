"""Whether a change request touches anything beyond the claimant's own corner of a graph."""

from __future__ import annotations

from typing import Any

__all__ = ["concerns"]

EDGE_OPS = {"add_edge", "remove_edge"}
REMOVE_OPS = {"remove", "remove_node"}


def concerns(request: dict[str, Any], edits: list[dict[str, Any]],
             graph: dict[str, Any]) -> list[str]:
    """Reasons the requested edits might not be the claimant's own information to change.

    ``request`` carries ``attested`` (they said the information is their own) and ``claimed``
    (the node they say they are, by id or label). An edit's ``target`` names a node by id or
    label, or an edge as ``a|rel|b``; an edge op may name one end in ``target`` and the other
    in ``other``. An attested claimant may remove any edge they are an end of, and any node
    nothing but them is joined to. Returns one sentence per concern; empty means none.
    """
    out = []
    if not request.get("attested"):
        out.append("did not say this is their own information")
    claimed = request.get("claimed") or ""
    if not claimed:
        out.append("did not say who they are")
        return out
    ids = {n["id"] for n in graph["nodes"]}
    label = {n["id"]: n["label"] for n in graph["nodes"]}
    named: dict[str, str] = {}
    for n in graph["nodes"]:
        named.setdefault(str(n["id"]).casefold(), n["id"])
        named.setdefault(str(n.get("label") or "").casefold(), n["id"])

    def resolve(x: Any) -> str:
        x = str(x or "")
        return x if x in ids else named.get(x.casefold(), x)

    claimed = resolve(claimed)
    if claimed not in ids:
        out.append(f"claims to be {claimed}, who is not in the graph")
        return out
    theirs = {claimed}
    for e in graph["edges"]:
        if e["source"] == claimed:
            theirs.add(e["target"])
        elif e["target"] == claimed:
            theirs.add(e["source"])
    for edit in edits:
        raw = str(edit.get("target", ""))
        parts = raw.split("|")
        a = b = ""
        if len(parts) == 3:
            a, b = resolve(parts[0]), resolve(parts[-1])
        elif edit.get("op") in EDGE_OPS:
            a, b = resolve(raw), resolve(edit.get("other", ""))
        if a or b:
            # a link is theirs only if they are one end of it; sharing the other end is not enough
            if claimed not in (a, b):
                out.append(f"asks to change the link {label.get(a, a)} -> {label.get(b, b)}, "
                           "which is not theirs")
            continue
        node = resolve(raw)
        neighbours: set[str] = set()
        others = 0
        for e in graph["edges"]:
            s, t = str(e["source"]), str(e["target"])
            far = t if s == node else s if t == node else None
            if far is None:
                continue
            neighbours.add(far)
            others += far not in (node, claimed)
        if edit.get("op") in REMOVE_OPS and node in ids:
            if not others:
                continue
            if node == claimed or claimed in neighbours:
                out.append(f"{label[node]} is shared with {others} other link(s) — "
                           "removing their own link to it is the change that is theirs")
            else:
                out.append(f"asks to change {label[node]}, which is not theirs")
            continue
        if node == claimed:
            continue
        if node not in theirs:
            out.append(f"asks to change {label.get(node, node)}, which is not theirs")
            continue
        if others:
            out.append(f"{label.get(node, node)} is shared with {others} other link(s)")
    return out
