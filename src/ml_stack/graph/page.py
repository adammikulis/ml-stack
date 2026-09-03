"""A graph as one page: force layout in two dimensions and three, on a map, searchable.

Every project that builds a graph then needs to look at it, and looking at it well is a lot of
work to redo — labels that do not collide, a legend that filters, a view that re-settles when a
kind is switched off, evidence for what is drawn. The page here does that for any graph shaped
as ``{"nodes": [...], "edges": [...]}``; what a project calls its kinds, and what it says about
them, are given rather than assumed.

Everything ships inside the file. There is no server behind it, no build step, and no request
made once it is open, so the page can be mailed, published, or served from a laptop and behaves
the same. That also sets the limit: the graph is in the file, so anyone who has the file has the
graph. For anything private, serve it rather than send it.

    html = render(graph, title="Who works on what",
                  kinds=[{"k": "person", "label": "People", "shape": "circle"},
                         {"k": "topic", "label": "Topics", "shape": "triangle"}])
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

WEB = Path(__file__).parent / "web"
SHAPES = ("circle", "square", "diamond", "triangle", "wye", "star", "cross")
# what a kind is drawn as, when the caller does not say
FALLBACK = ("circle", "square", "diamond", "triangle", "wye", "star", "cross")


def hidden(node: Mapping[str, Any]) -> bool:
    """A node kept for the record rather than the map -- an ingest run, a unit -- carries
    ``attrs.hidden``; the page and the listing tools leave it out, a question about origin
    can still reach it through the store."""
    return bool((node.get("attrs") or {}).get("hidden"))


def shown(graph: Mapping[str, Any]) -> dict[str, Any]:
    """The graph without its hidden nodes and the edges that touch them."""
    nodes = [n for n in graph.get("nodes") or () if not hidden(n)]
    ids = {n.get("id") for n in nodes}
    edges = [e for e in graph.get("edges") or ()
             if e.get("source") in ids and e.get("target") in ids]
    return {**graph, "nodes": nodes, "edges": edges}


def kinds_of(graph: Mapping[str, Any]) -> list[dict[str, str]]:
    """One entry per kind the graph actually holds, in the order they first appear --
    hidden nodes' kinds left out."""
    seen: list[str] = []
    for node in graph.get("nodes") or ():
        kind = str(node.get("kind") or "")
        if kind and kind not in seen and not hidden(node):
            seen.append(kind)
    return [{"k": kind, "label": kind.replace("_", " ").title() + "s",
             "shape": FALLBACK[i % len(FALLBACK)]} for i, kind in enumerate(seen)]


def world_outline() -> dict[str, Any]:
    """The land, as topojson, for a page that places things geographically."""
    raw = json.loads((WEB / "countries-50m.json").read_text(encoding="utf-8"))
    return {"type": "Topology", "objects": {"land": raw["objects"]["land"]},
            "arcs": raw["arcs"], "transform": raw.get("transform")}


def _embedded(obj: Any) -> str:
    """JSON safe to put inside a <script> tag."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render(graph: Mapping[str, Any], *, title: str = "Graph", brand: str = "",
           kinds: Sequence[Mapping[str, Any]] | None = None,
           copy: Mapping[str, str] | None = None,
           points: Sequence[Mapping[str, Any]] = (),
           world: Mapping[str, Any] | None = None,
           author: str = "", extra: Mapping[str, Any] | None = None) -> str:
    """The whole page, as one string.

    ``brand`` names whatever made the page, on the bar above it; ``title`` names the graph,
    over the graph itself. ``points`` are ``{id, label, place, lat, lon}`` for anything to show
    on the map; passing none leaves the map empty. ``extra`` is merged into the payload the page reads, for
    whatever a caller's own panels need.
    """
    template = (WEB / "graph.html").read_text(encoding="utf-8")
    payload = {"title": title, "graph": shown(graph), "points": list(points),
               "kinds": list(kinds) if kinds is not None else kinds_of(graph),
               "copy": dict(copy or {}), "author": author, **dict(extra or {})}
    return (template
            .replace("__BRAND__", brand or title)
            .replace("__TITLE__", title)
            .replace("__DATA__", _embedded(payload))
            .replace("__WORLD__", _embedded(world if world is not None else world_outline())))
