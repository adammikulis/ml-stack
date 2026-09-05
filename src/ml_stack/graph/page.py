"""A graph as one page: force layout in two dimensions and three, on a map, searchable.

The page is assembled from components (`ml_stack.ui`): one file each under ``web/components``,
a custom element apiece, sharing one model (``page-model``). ``COMPONENTS`` is the page; a
caller may hand ``render`` a shorter list, or a longer one with components of its own. The
graph is ``{"nodes": [...], "edges": [...]}``; what a project calls its kinds, and what it
says about them, are given rather than assumed.

Everything ships inside the file: no build step, no request made once it is open, so the
page can be mailed, published, or served from a laptop and behaves the same. Anyone who has
the file has the graph; for anything private, serve it rather than send it.

    html = render(graph, title="Who works on what",
                  kinds=[{"k": "person", "label": "People", "shape": "circle"},
                         {"k": "topic", "label": "Topics", "shape": "triangle"}])
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ml_stack.ui import Component, assemble, load

WEB = Path(__file__).parent / "web"
COMPONENTS_DIR = WEB / "components"
#: the page, in the order the elements wire themselves up
COMPONENTS = ("page-model", "graph-banner", "graph-stats", "graph-search", "graph-view",
              "graph-grips", "graph-map", "graph-history", "graph-3d", "graph-detail",
              "ask-pane", "draft-note", "change-request", "display-panel", "refresh-button",
              "review-queue")
SHAPES = ("circle", "square", "diamond", "triangle", "wye", "star", "cross")
# what a kind is drawn as, when the caller does not say
FALLBACK = ("circle", "square", "diamond", "triangle", "wye", "star", "cross")
# the five kinds the shell paints itself, light and dark, in its own stylesheet
SHIPPED = {"person": "#2a78d6", "org": "#eb6834", "place": "#1baf7a", "topic": "#eda100",
           "opportunity": "#e87ba4"}
# what any other kind is painted, by position, when the caller does not say
PALETTE = ("#7a5af5", "#0e9aa7", "#b833a6", "#8a9a1b", "#d63a3a", "#4b4fbf", "#a0522d",
           "#6b7a8f", "#2f9e44", "#c2410c")


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


def coloured(kinds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each kind entry with a ``colour``: the one given, a shipped kind's own, or the next
    of ``PALETTE`` for anything else."""
    out: list[dict[str, Any]] = []
    taken = {str(e.get("colour")).lower() for e in kinds if e.get("colour")}
    free = [c for c in PALETTE if c not in taken] or list(PALETTE)
    chosen = 0
    for entry in kinds:
        entry = dict(entry)
        kind = str(entry.get("k") or "")
        if not entry.get("colour"):
            if kind in SHIPPED:
                entry["colour"] = SHIPPED[kind]
            else:
                entry["colour"] = free[chosen % len(free)]
                chosen += 1
        out.append(entry)
    return out


def kind_style(kinds: Sequence[Mapping[str, Any]]) -> str:
    """A ``<style>`` painting every kind that the shell does not: a given colour, and the
    palette colour of a kind the shell never heard of. Empty when there is none."""
    rules = [f"--k-{k['k']}:{k['colour']}" for k in kinds
             if k.get("k") and k.get("colour") and k["colour"] != SHIPPED.get(k["k"])]
    if not rules:
        return ""
    return ('<style id="kind-colours">:root, :root:not([data-theme="light"]), '
            ':root[data-theme="dark"] {' + ";".join(rules) + '}</style>')


def newest(graph: Mapping[str, Any], most: int) -> tuple[dict[str, Any], int]:
    """The graph with only its ``most`` newest messages by ``ts``, and how many went."""
    held = dict(graph.get("messages") or {})
    if len(held) <= most:
        return dict(graph), 0
    order = sorted(held, key=lambda i: float((held[i] or {}).get("ts") or 0), reverse=True)
    kept = set(order[:max(0, most)])
    trimmed = {**graph, "messages": {i: held[i] for i in order[:max(0, most)]}}
    for key in ("nodes", "edges"):
        trimmed[key] = [{**one, "messages": [m for m in one.get("messages") or () if m in kept]}
                        if "messages" in one else dict(one) for one in graph.get(key) or ()]
    return trimmed, len(held) - len(kept)


def world_outline() -> dict[str, Any]:
    """The land, as topojson, for a page that places things geographically."""
    raw = json.loads((WEB / "countries-50m.json").read_text(encoding="utf-8"))
    return {"type": "Topology", "objects": {"land": raw["objects"]["land"]},
            "arcs": raw["arcs"], "transform": raw.get("transform")}


def _embedded(obj: Any) -> str:
    """JSON safe to put inside a <script> tag."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def components(names: Sequence[str | Component] = COMPONENTS) -> list[Component]:
    """The named components: a bare name is one of the page's own under ``web/components``;
    a `Component` is taken as given, wherever its file lives."""
    return [c if isinstance(c, Component) else load(COMPONENTS_DIR, [c])[0] for c in names]


def template(names: Sequence[str | Component] = COMPONENTS) -> str:
    """The page with its components in place and the payload markers still in it."""
    shell = (WEB / "shell.html").read_text(encoding="utf-8")
    return assemble(shell, components(names))


def render(graph: Mapping[str, Any], *, title: str = "Graph", brand: str = "",
           kinds: Sequence[Mapping[str, Any]] | None = None,
           copy: Mapping[str, str] | None = None,
           points: Sequence[Mapping[str, Any]] | None = None,
           world: Mapping[str, Any] | None = None,
           author: str = "", extra: Mapping[str, Any] | None = None,
           most_messages: int | None = None,
           parts: Sequence[str | Component] = COMPONENTS) -> str:
    """The whole page, as one string.

    ``brand`` names whatever made the page, on the bar above it; ``title`` names the graph,
    over the graph itself. ``points`` are ``{id, label, place, lat, lon}`` for anything to
    show on the map, and default to every node whose attributes carry ``lat`` and ``lon``
    (`graph.geocode` writes them); ``points=()`` leaves the map empty. ``extra`` is merged into the payload
    the page reads, for whatever a caller's own panels need. A kind entry may carry
    ``colour`` (a hex string); one without gets a colour of its own, so no kind paints black.
    ``most_messages`` keeps only that many of the newest messages, and the payload's
    ``messagesLeftOut`` says how many did not fit. ``parts`` is the page's components,
    `COMPONENTS` unless a caller leaves some out or adds its own.
    """
    from ml_stack.graph.places import points as placed

    page = template(parts)
    visible, left_out = shown(graph), 0
    if most_messages is not None:
        visible, left_out = newest(visible, int(most_messages))
    kinds = coloured(list(kinds) if kinds is not None else kinds_of(graph))
    payload = {"title": title, "graph": visible,
               "points": placed(visible) if points is None else list(points),
               "kinds": kinds, "messagesLeftOut": left_out,
               "copy": dict(copy or {}), "author": author, **dict(extra or {})}
    return (page
            .replace("</style>", "</style>\n" + kind_style(kinds), 1)
            .replace("__BRAND__", brand or title)
            .replace("__TITLE__", title)
            .replace("__DATA__", _embedded(payload))
            .replace("__WORLD__", _embedded(world if world is not None else world_outline())))
