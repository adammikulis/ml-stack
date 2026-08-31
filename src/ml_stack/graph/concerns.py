"""Whether a change request touches anything beyond the claimant's own corner of a graph."""

from __future__ import annotations

from typing import Any

__all__ = ["concerns"]


def concerns(request: dict[str, Any], edits: list[dict[str, Any]],
             graph: dict[str, Any]) -> list[str]:
    """Reasons the requested edits might not be the claimant's own information to change.

    ``request`` carries ``attested`` (they said the information is their own) and ``claimed``
    (the node id they say they are). An edit's ``target`` is a node id, or ``a|rel|b`` for an
    edge. Returns one sentence per concern; empty means none.
    """
    out = []
    if not request.get("attested"):
        out.append("did not say this is their own information")
    claimed = request.get("claimed") or ""
    if not claimed:
        out.append("did not say who they are")
        return out
    if not any(n["id"] == claimed for n in graph["nodes"]):
        out.append(f"claims to be {claimed}, who is not in the graph")
        return out
    theirs = {claimed}
    for e in graph["edges"]:
        if e["source"] == claimed:
            theirs.add(e["target"])
        elif e["target"] == claimed:
            theirs.add(e["source"])
    label = {n["id"]: n["label"] for n in graph["nodes"]}
    for edit in edits:
        target = edit.get("target", "")
        parts = target.split("|")
        if len(parts) == 3:
            # a link is theirs only if they are one end of it; sharing the other end is not enough
            if claimed not in {parts[0], parts[-1]}:
                a, b = label.get(parts[0], parts[0]), label.get(parts[-1], parts[-1])
                out.append(f"asks to change the link {a} -> {b}, which is not theirs")
            continue
        if target == claimed:
            continue
        if target not in theirs:
            out.append(f"asks to change {label.get(target, target)}, which is not theirs")
            continue
        others = sum(1 for e in graph["edges"]
                     if target in (e["source"], e["target"])
                     and claimed not in (e["source"], e["target"]))
        if others:
            out.append(f"{label.get(target, target)} is shared with {others} other link(s)")
    return out
