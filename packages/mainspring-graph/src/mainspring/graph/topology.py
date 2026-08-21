"""Building a graph from point features, rather than being handed one.

When a model constructs its own topology each forward pass, edge construction sits in the
inner loop and its determinism matters as much as its speed: two backends must produce the
*same* edges, or a numerical parity test fails for a reason that has nothing to do with the
arithmetic.

Both constructions here are exactly reproducible:

* **kNN** is a full pairwise distance argsort. Ties are broken by index because ``argsort``
  is stable, so the edge set is a pure function of the coordinates.
* **MST on a point cloud is unique** when the pairwise distances are distinct, which they
  are for float coordinates in general position. Coincident points are the exception, and
  they are handled explicitly below.

The distance matrix is O(N²), which is the right trade below a few thousand nodes and the
wrong one above it. There is no approximate path here; if you need one, that is a real
addition rather than a tuning knob.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Tensor = Any


@dataclass(frozen=True, slots=True)
class Edges:
    """An edge set as host-side index arrays, ready to become a ``Graph``."""

    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray

    def __len__(self) -> int:
        return int(self.src.shape[0])

    def symmetrized(self) -> "Edges":
        """Both directions for every edge, so an undirected graph propagates both ways."""
        return Edges(
            src=np.concatenate([self.src, self.dst]),
            dst=np.concatenate([self.dst, self.src]),
            weight=np.concatenate([self.weight, self.weight]),
        )


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Full Euclidean distance matrix, ``(N, N)``.

    Computed from the squared-norm expansion and then clipped at zero: the expansion is
    fast but can produce a small negative value on the diagonal through cancellation, and
    ``sqrt`` of that is NaN.
    """
    points = np.asarray(points, dtype=np.float64)
    squared = np.sum(points**2, axis=1)
    d2 = squared[:, None] + squared[None, :] - 2.0 * (points @ points.T)
    return np.sqrt(np.maximum(d2, 0.0))


def knn_edges(points: np.ndarray, k: int) -> Edges:
    """Each node to its ``k`` nearest neighbours, excluding itself.

    Deterministic: ``argsort`` is stable, so equidistant neighbours are taken in index
    order rather than in whatever order the sort happened to produce.
    """
    distances = pairwise_distances(points)
    n = distances.shape[0]
    if n <= 1 or k <= 0:
        return _empty_edges()

    np.fill_diagonal(distances, np.inf)  # never a neighbour of itself
    k = min(int(k), n - 1)
    neighbours = np.argsort(distances, axis=1, kind="stable")[:, :k]

    src = np.repeat(np.arange(n), k)
    dst = neighbours.reshape(-1)
    return Edges(src=src, dst=dst, weight=distances[src, dst])


def mst_edges(points: np.ndarray) -> Edges:
    """Minimum spanning tree over the point cloud, by Prim's algorithm.

    Coincident points need care. A distance of exactly zero is indistinguishable from
    "no edge" in a sparse representation, and duplicate points are common in real feature
    data -- two tokens with identical embeddings, a padded row. Zero distances are nudged
    to the smallest representable positive float so the pair is still connected by an edge
    of negligible weight rather than being silently dropped.

    Prim from node 0 rather than Kruskal: the output order is then a pure function of the
    coordinates, which is what makes two backends produce byte-identical edge lists.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n <= 1:
        return _empty_edges()

    distances = pairwise_distances(points)
    off_diagonal = ~np.eye(n, dtype=bool)
    coincident = (distances == 0.0) & off_diagonal
    if coincident.any():
        distances = distances.copy()
        distances[coincident] = np.nextafter(0.0, 1.0)

    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best_cost = distances[0].copy()
    best_from = np.zeros(n, dtype=np.int64)
    best_cost[0] = np.inf

    src = np.empty(n - 1, dtype=np.int64)
    dst = np.empty(n - 1, dtype=np.int64)
    weight = np.empty(n - 1, dtype=np.float64)

    for i in range(n - 1):
        candidate = int(np.argmin(np.where(in_tree, np.inf, best_cost)))
        src[i] = best_from[candidate]
        dst[i] = candidate
        weight[i] = best_cost[candidate]
        in_tree[candidate] = True

        closer = distances[candidate] < best_cost
        best_from = np.where(closer, candidate, best_from)
        best_cost = np.where(closer, distances[candidate], best_cost)

    return Edges(src=src, dst=dst, weight=weight)


def morton_codes(points: np.ndarray, *, bits: int = 10) -> np.ndarray:
    """Z-order curve index per point, for spatial locality.

    Exact integer bit arithmetic on quantised coordinates, so it is identical on every
    backend and every platform -- no floating-point comparison anywhere in the result.

    Points are normalised to the unit cube first, so the codes describe relative position
    within this cloud rather than absolute coordinates.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    lo = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - lo, np.finfo(np.float64).tiny)
    scale = (1 << bits) - 1
    quantised = np.clip(((points - lo) / span * scale).astype(np.int64), 0, scale)

    codes = np.zeros(points.shape[0], dtype=np.int64)
    for dim in range(quantised.shape[1]):
        codes |= _interleave(quantised[:, dim], quantised.shape[1], bits) << dim
    return codes


def _interleave(values: np.ndarray, stride: int, bits: int) -> np.ndarray:
    """Spread each bit of ``values`` ``stride`` positions apart."""
    out = np.zeros_like(values)
    for bit in range(bits):
        out |= ((values >> bit) & 1) << (bit * stride)
    return out


def spatial_window_edges(points: np.ndarray, window: int) -> Edges:
    """Edges between points that are near each other along the Morton curve.

    A cheap locality prior: sort by Z-order and connect each point to the next ``window``
    in that order. Costs O(N log N) against kNN's O(N²), and gives up exactness for it --
    the Z-order curve puts spatially close points close together *usually*, not always.
    """
    codes = morton_codes(points)
    order = np.argsort(codes, kind="stable")
    n = order.shape[0]
    if n <= 1 or window <= 0:
        return _empty_edges()

    src_list: list[int] = []
    dst_list: list[int] = []
    for offset in range(1, min(int(window), n - 1) + 1):
        src_list.extend(order[:-offset].tolist())
        dst_list.extend(order[offset:].tolist())

    src = np.asarray(src_list, dtype=np.int64)
    dst = np.asarray(dst_list, dtype=np.int64)
    distances = np.linalg.norm(
        np.asarray(points, dtype=np.float64)[src] - np.asarray(points, dtype=np.float64)[dst],
        axis=1,
    )
    return Edges(src=src, dst=dst, weight=distances)


def build_topology(
    points: np.ndarray,
    *,
    k: int = 4,
    include_mst: bool = True,
    window: int = 0,
    symmetric: bool = True,
) -> Edges:
    """kNN, optionally plus an MST, optionally plus a Morton window. Deduplicated.

    The MST is worth including because kNN alone can leave a graph disconnected -- a
    cluster of points whose k nearest neighbours are all inside the cluster has no edge
    out of it, and no amount of message passing will ever move information across that
    gap. The MST costs N-1 edges and guarantees one connected component.
    """
    parts = [knn_edges(points, k)] if k > 0 else []
    if include_mst:
        parts.append(mst_edges(points))
    if window > 0:
        parts.append(spatial_window_edges(points, window))

    if not parts:
        return _empty_edges()

    combined = Edges(
        src=np.concatenate([p.src for p in parts]),
        dst=np.concatenate([p.dst for p in parts]),
        weight=np.concatenate([p.weight for p in parts]),
    )
    if symmetric:
        combined = combined.symmetrized()
    return _dedupe(combined)


def _dedupe(edges: Edges) -> Edges:
    """Drop duplicate (src, dst) pairs and self-loops, keeping the first weight seen."""
    if len(edges) == 0:
        return edges
    keep = edges.src != edges.dst
    src, dst, weight = edges.src[keep], edges.dst[keep], edges.weight[keep]

    pairs = np.stack([src, dst], axis=1)
    _unique, index = np.unique(pairs, axis=0, return_index=True)
    order = np.sort(index)  # np.unique sorts; restore construction order for stability
    return Edges(src=src[order], dst=dst[order], weight=weight[order])


def _empty_edges() -> Edges:
    return Edges(
        src=np.empty(0, dtype=np.int64),
        dst=np.empty(0, dtype=np.int64),
        weight=np.empty(0, dtype=np.float64),
    )
