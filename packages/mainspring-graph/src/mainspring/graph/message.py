"""Moving values along edges: the one thing a graph library has to do on-device.

Everything here is differentiable and stays in the array backend. Nothing here calls
``.tolist()`` -- that would sync the device to the host on every step, which for a graph
rebuilt each forward pass is the whole cost of the model.
"""

from __future__ import annotations

from typing import Any

from mainspring.graph.data import Graph

Tensor = Any


def gather(backend: Any, x: Tensor, index: Tensor) -> Tensor:
    """``x[index]`` along the node axis. The 'send' half of message passing."""
    return backend.ops.take(x, index, axis=0)


def scatter_sum(backend: Any, values: Tensor, index: Tensor, num_nodes: int) -> Tensor:
    """Sum ``values`` into ``num_nodes`` slots by ``index``. The 'receive' half."""
    return backend.segment_sum(values, index, int(num_nodes), 0)


def scatter_mean(backend: Any, values: Tensor, index: Tensor, num_nodes: int) -> Tensor:
    """Mean instead of sum, with empty slots left at zero rather than NaN.

    A node with no in-edges divides by a count of zero. Clamping the denominator to 1
    yields 0 for that node, which is what a mean over nothing should contribute -- whereas
    NaN would propagate through the rest of the layer and destroy the whole batch.
    """
    ops = backend.ops
    totals = scatter_sum(backend, values, index, num_nodes)
    counts = scatter_sum(backend, ops.ones_like(values), index, num_nodes)
    return totals / ops.maximum(counts, ops.ones_like(counts))


def propagate(
    backend: Any,
    graph: Graph,
    x: Tensor,
    *,
    reduce: str = "sum",
    use_weights: bool = True,
) -> Tensor:
    """One round of message passing: gather from ``src``, scatter into ``dst``.

    ``x`` is ``(num_nodes, d)``; the result has the same shape. This is the primitive a
    GCN/GraphSAGE layer is built from -- the normalisation and the learned projection are
    the caller's business, deliberately, because every architecture spells them
    differently.
    """
    messages = gather(backend, x, graph.src)

    if use_weights and graph.w is not None:
        weights = backend.ops.reshape(graph.w, (graph.num_edges,) + (1,) * (len(x.shape) - 1))
        messages = messages * weights

    if reduce == "sum":
        return scatter_sum(backend, messages, graph.dst, graph.num_nodes)
    if reduce == "mean":
        return scatter_mean(backend, messages, graph.dst, graph.num_nodes)
    raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")


def degree(backend: Any, graph: Graph, *, incoming: bool = True) -> Tensor:
    """In- or out-degree per node, as a float array."""
    ops = backend.ops
    index = graph.dst if incoming else graph.src
    ones = ops.ones((graph.num_edges,), dtype=ops.float32)
    return scatter_sum(backend, ones, index, graph.num_nodes)


def normalize_by_degree(backend: Any, graph: Graph, x: Tensor) -> Tensor:
    """Symmetric normalisation, the ``D^-1/2 A D^-1/2`` of a GCN layer.

    Isolated nodes get a degree of zero; their normalisation factor is clamped to 1 rather
    than producing an infinity from ``rsqrt(0)``.
    """
    ops = backend.ops
    deg = degree(backend, graph, incoming=True)
    inv_sqrt = ops.rsqrt(ops.maximum(deg, ops.ones_like(deg)))
    return x * ops.reshape(inv_sqrt, (graph.num_nodes,) + (1,) * (len(x.shape) - 1))
