"""What a graph means, as vectors a reader can search.

The word index finds a name typed nearly right; vectors find a thing described another way
— "who fixes machines" reaching a robotics technician who never wrote either word. Both are
built here while there is write access, because a reader has none.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ml_stack.client.embed import BATCH, DOCUMENT

MOST_CHARS = 1400


def texts_for(graph: Mapping[str, Any], *, most: int = MOST_CHARS) -> dict[str, str]:
    """Node id -> its label and the sentences it was read from, capped at ``most`` characters.

    A node with neither a label nor anything said maps to ``""``, not to the separator alone.
    """
    messages = graph.get("messages") or {}
    out: dict[str, str] = {}
    for node in graph.get("nodes") or ():
        label = str(node.get("label", ""))
        said = " ".join((messages.get(m) or {}).get("text", "")
                        for m in (node.get("messages") or ()))
        text = f"{label} — {said}" if said else label
        out[str(node["id"])] = text[:most]
    return out


def to_embed(texts: Mapping[str, str]) -> list[str]:
    """Node ids, out of ``texts``, with something worth embedding."""
    return [i for i in sorted(texts) if str(texts[i]).strip()]


def coverage(store: Any, *, model: str = "") -> tuple[int, int]:
    """(nodes with a vector, nodes with something to embed), for ``model``."""
    return embedded(store, model=model), len(to_embed(texts_for(store.read())))


@dataclass
class RememberOptions:
    """The less-common knobs of :func:`remember`, grouped off its own signature."""

    batch: int = BATCH
    embedder: Callable[..., list[list[float]]] | None = None
    smooth_hops: int = 0
    graph: Mapping[str, Any] | None = None
    log: Callable[[str], None] | None = None


def remember(store: Any, texts: Mapping[str, str], *, base_url: str, model: str,
             options: RememberOptions | None = None) -> int:
    """Embed what each node was read from, and store it against that node.

    ``texts`` is node id -> the words that stand for it, carried with :data:`DOCUMENT`; a
    question compared against these embeds with :data:`QUERY` instead. ``options.smooth_hops``
    above 0 spreads the result with :func:`smooth`, over ``options.graph`` or the store's own.

    Returns how many nodes hold a vector afterwards; a batch the server refuses is skipped.
    """
    opts = options or RememberOptions()
    embedder = opts.embedder
    if embedder is None:
        from ml_stack.client.embed import embed as embedder
    ids = to_embed(texts)
    written = 0
    made: dict[str, list[float]] = {}
    for at in range(0, len(ids), max(1, opts.batch)):
        chunk = ids[at:at + max(1, opts.batch)]
        try:
            vectors = embedder([DOCUMENT + texts[i] for i in chunk],
                               base_url=base_url, model=model, timeout=300)
        except Exception as exc:  # noqa: BLE001 - one bad batch is not the whole graph
            if opts.log:
                opts.log(f"vectors: {len(chunk)} could not be embedded: {exc}")
            continue
        for node_id, vector in zip(chunk, vectors, strict=False):
            store.set_embedding(node_id, vector, model=model)
            made[node_id] = [float(x) for x in vector]
            written += 1
    if opts.log:
        opts.log(f"vectors: {written} of {len(ids)} nodes embedded with {model}")
    if int(opts.smooth_hops) > 0 and made:
        held = opts.graph if opts.graph is not None else store.read()
        spread = smooth(held, made, hops=int(opts.smooth_hops))
        for node_id, vector in spread.items():
            store.set_embedding(node_id, vector, model=model)
        written = len(spread)
        if opts.log:
            opts.log(f"vectors: {written} smoothed over {opts.smooth_hops} hop(s)")
    return written


def embedded(store: Any, *, model: str = "") -> int:
    """How many vectors the store holds, for the model that built them."""
    try:
        rows = store.query(
            "MATCH (e:Embedding) WHERE $m = '' OR e.model = $m RETURN count(e) AS n",
            {"m": str(model)})
    except RuntimeError:
        return 0                      # no Embedding table yet, which is none of them
    return int(rows[0]["n"]) if rows else 0


def smooth(graph: Mapping[str, Any], vectors: Mapping[str, Sequence[float]], *,
           hops: int = 2, backend: Any = None) -> dict[str, list[float]]:
    """Every node's vector spread over its neighbourhood, as ``{id: vector}``.

    ``hops`` rounds of ``D^-1/2 (A + I) D^-1/2`` over the graph read as undirected, each
    result scaled to unit length. A node with no vector of its own starts at zero and ends
    with what its neighbours mean; one still at zero is left out.
    """
    from ml_stack.backend import get_backend
    from ml_stack.graph.data import Graph, _to_list
    from ml_stack.graph.message import normalize_by_degree, propagate

    held = {str(i): [float(x) for x in v] for i, v in vectors.items() if v is not None}
    ids = [str(n.get("id")) for n in graph.get("nodes") or () if n.get("id") is not None]
    if not held or not ids:
        return {}
    width = len(next(iter(held.values())))
    at = {node_id: i for i, node_id in enumerate(ids)}

    src = list(range(len(ids)))                       # the I of A + I
    dst = list(range(len(ids)))
    for edge in graph.get("edges") or ():
        a, b = at.get(str(edge.get("source"))), at.get(str(edge.get("target")))
        if a is None or b is None:
            continue
        src += [a, b]
        dst += [b, a]

    backend = backend or get_backend()
    ops = backend.ops
    joined = Graph(num_nodes=len(ids), src=ops.array(src, dtype=ops.int32),
                   dst=ops.array(dst, dtype=ops.int32))
    x = ops.array([held.get(i) or [0.0] * width for i in ids], dtype=ops.float32)
    for _ in range(max(0, int(hops))):
        x = normalize_by_degree(backend, joined, x)
        x = propagate(backend, joined, x, reduce="sum")
        x = normalize_by_degree(backend, joined, x)

    out: dict[str, list[float]] = {}
    for node_id, row in zip(ids, _to_list(x), strict=True):
        size = sum(v * v for v in row) ** 0.5
        if size > 0:
            out[node_id] = [v / size for v in row]
    return out
