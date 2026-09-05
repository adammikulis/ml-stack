"""Where a graph's entries are: a place name given a point, and edges between the nearest."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ml_stack import geo

__all__ = ["EARTH_KM", "geocode", "kilometres", "places_in", "points"]

EARTH_KM = 6371.0088


def places_in(graph: Mapping[str, Any]) -> dict[str, str]:
    """``{node id: the place it names}`` -- an ``attrs.place``, or a ``place`` node's label."""
    out: dict[str, str] = {}
    for node in graph.get("nodes") or ():
        node_id = str(node.get("id") or "")
        attrs = node.get("attrs") or {}
        said = str(attrs.get("place") or "").strip()
        if not said and str(node.get("kind") or "") == "place":
            said = str(node.get("label") or "").strip()
        if node_id and said:
            out[node_id] = said
    return out


def geocode(graph: Mapping[str, Any], cache_path: str | Path, *, near: int = 0,
            lookup: Callable[..., dict[str, Any] | None] | None = None,
            log: Callable[[str], None] = print) -> dict[str, Any]:
    """The graph with ``lat`` and ``lon`` on every node whose place was found.

    Each distinct place goes through :func:`ml_stack.geo.geocode_all`, so the JSON cache at
    ``cache_path`` is asked before Nominatim is. ``near`` above 0 adds a ``near`` edge from
    each placed node to its ``near`` closest, weighted ``1 / (1 + kilometres)``, replacing
    the ``near`` edges the graph already carried. ``lookup`` answers one place,
    :func:`ml_stack.geo.lookup` unless a caller has its own.
    """
    wanted = places_in(graph)
    found = geo.geocode_all(sorted(set(wanted.values())), Path(cache_path),
                            ask=lookup or geo.lookup, log=log)

    nodes: list[dict[str, Any]] = []
    placed: list[tuple[str, float, float]] = []
    for node in graph.get("nodes") or ():
        said = wanted.get(str(node.get("id") or ""))
        spot = found.get(said) if said else None
        if not isinstance(spot, Mapping):
            nodes.append(dict(node))
            continue
        lat, lon = float(spot["lat"]), float(spot["lon"])
        nodes.append({**node, "attrs": {**(node.get("attrs") or {}), "place": said,
                                       "lat": lat, "lon": lon}})
        placed.append((str(node["id"]), lat, lon))

    edges = [dict(e) for e in graph.get("edges") or ()
             if not (near > 0 and str(e.get("rel") or "") == "near")]
    if near > 0 and len(placed) > 1:
        edges += _near_edges(placed, near)
    return {**graph, "nodes": nodes, "edges": edges}


def points(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``{id, label, place, lat, lon}`` for every node carrying a point, for the map."""
    out: list[dict[str, Any]] = []
    for node in graph.get("nodes") or ():
        attrs = node.get("attrs") or {}
        if attrs.get("hidden") or attrs.get("lat") is None or attrs.get("lon") is None:
            continue
        out.append({"id": str(node.get("id") or ""),
                    "label": str(node.get("label") or node.get("id") or ""),
                    "place": str(attrs.get("place") or ""),
                    "lat": float(attrs["lat"]), "lon": float(attrs["lon"])})
    return out


def kilometres(one: tuple[float, float], other: tuple[float, float]) -> float:
    """Great-circle distance between two ``(lat, lon)`` pairs."""
    lat1, lon1 = math.radians(one[0]), math.radians(one[1])
    lat2, lon2 = math.radians(other[0]), math.radians(other[1])
    under = (math.sin((lat2 - lat1) / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(under)))


def _near_edges(placed: list[tuple[str, float, float]], near: int) -> list[dict[str, Any]]:
    """One ``near`` edge per unordered pair the kNN over the unit sphere found."""
    import numpy as np

    from ml_stack.graph.topology import knn_edges

    on_sphere = np.array([_unit(lat, lon) for _, lat, lon in placed], dtype=np.float64)
    close = knn_edges(on_sphere, int(near))
    made: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for u, v in zip(close.src.tolist(), close.dst.tolist()):
        a, b = placed[int(u)], placed[int(v)]
        pair = (min(a[0], b[0]), max(a[0], b[0]))
        if a[0] == b[0] or pair in seen:
            continue
        seen.add(pair)
        km = kilometres((a[1], a[2]), (b[1], b[2]))
        made.append({"source": a[0], "target": b[0], "rel": "near",
                     "weight": round(1.0 / (1.0 + km), 6)})
    return made


def _unit(lat: float, lon: float) -> tuple[float, float, float]:
    """A latitude and longitude as a point on the unit sphere."""
    phi, lam = math.radians(lat), math.radians(lon)
    return (math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi))
